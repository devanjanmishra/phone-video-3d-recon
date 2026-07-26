"""
Stage 2: Keyframes -> camera poses + dense metric depth -> fused point cloud.

Backbone: Depth Anything 3 (ByteDance Seed, arXiv 2511.10647) - current SOTA
for pose + geometry from unposed images/video. The NESTED model outputs
depth already in METERS, which is what makes downstream CAD extraction
meaningful.

Outputs (in <workdir>/recon):
  recon.npz          poses (w2c 3x4), intrinsics, depth, confidence, image list
  fused_points.ply   confidence-filtered, voxel-downsampled world-space cloud
  colmap/sparse/0/   cameras.txt / images.txt / points3D.txt  (for gsplat/3DGS)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("video2cad.da3")

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"  # metric scale, CC BY-NC
APACHE_MODEL = "depth-anything/DA3-BASE"                     # permissive, lower accuracy


def _pick_dtype(torch, device) -> "torch.dtype":
    """bf16 needs Ampere (sm_80+). Pre-Ampere falls back to fp32 because
    DA3's layer_norm raises 'expected Float but found Half' with fp16 on
    older torch builds (e.g. 2.5.x)."""
    if device.type != "cuda":
        return torch.float32
    major, minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return torch.bfloat16
    log.info("GPU is sm_%d%d (pre-Ampere) - using fp32 (fp16 not supported by DA3 layer_norm)", major, minor)
    return torch.float32


def run_da3(
    frame_paths: list[Path],
    workdir: Path,
    model_dir: str = DEFAULT_MODEL,
    conf_percentile: float = 40.0,
    voxel_size: float = 0.01,
    max_depth: float = 12.0,
    use_ray_pose: bool = True,
    batch_limit: int = 160,
    device_str: str = "auto",
    dtype_str: str = "auto",
    process_res: int = 504,
) -> dict:
    """Run DA3 over keyframes and fuse a world-space metric point cloud."""
    import torch
    from depth_anything_3.api import DepthAnything3

    recon_dir = workdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    if len(frame_paths) > batch_limit:
        log.info(
            "%d frames > batch_limit=%d. Will process in overlapping chunks.",
            len(frame_paths), batch_limit,
        )

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if device.type != "cuda":
        log.warning(
            "Running on CPU - expect ~20-60 min for 30 frames. If you have an "
            "NVIDIA GPU, your torch is likely the CPU-only wheel; reinstall with "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    else:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(0), vram)
        if vram < 8 and "GIANT" in model_dir.upper():
            raise RuntimeError(
                f"{model_dir} needs ~24GB VRAM; you have {vram:.1f}GB. "
                "Use --model depth-anything/DA3-SMALL (or DA3-BASE) instead. "
                "Note: only the NESTED models emit metric depth - with SMALL/BASE "
                "the cloud is up-to-scale and must be rescaled from one known "
                "measurement before the DXF dimensions are trustworthy."
            )

    dtype = _pick_dtype(torch, device) if dtype_str == "auto" else getattr(torch, dtype_str)
    log.info("Loading DA3 model: %s (dtype=%s)", model_dir, dtype)
    model = DepthAnything3.from_pretrained(model_dir).to(device=device, dtype=dtype)
    model.eval()

    images = [str(p) for p in frame_paths]

    # Decide: single pass or batched with overlap
    if len(images) <= batch_limit:
        log.info("Running DA3 inference on %d frames (ray pose=%s, process_res=%d)...",
                 len(images), use_ray_pose, process_res)
        depth, conf, w2c, K, imgs = _infer_single(
            model, images, use_ray_pose, process_res)
    else:
        log.info("Running DA3 batched inference: %d frames in chunks of %d "
                 "(overlap=%d, process_res=%d)...",
                 len(images), batch_limit, batch_limit // 2, process_res)
        depth, conf, w2c, K, imgs = _infer_batched(
            model, images, batch_limit, use_ray_pose, process_res)

    np.savez_compressed(
        recon_dir / "recon.npz",
        depth=depth, conf=conf, extrinsics=w2c, intrinsics=K,
        images=np.array(images),
    )

    pcd_path = fuse_point_cloud(
        imgs, depth, conf, w2c, K, recon_dir,
        conf_percentile=conf_percentile, voxel_size=voxel_size, max_depth=max_depth,
    )

    colmap_dir = export_colmap(imgs, w2c, K, frame_paths, recon_dir)

    return {
        "npz": recon_dir / "recon.npz",
        "point_cloud": pcd_path,
        "colmap": colmap_dir,
        "n_frames": len(images),
    }


def _infer_single(model, images, use_ray_pose, process_res):
    """Single-pass inference (all frames fit in VRAM)."""
    import torch

    try:
        with torch.inference_mode():
            try:
                pred = model.inference(images, use_ray_pose=use_ray_pose,
                                       process_res=process_res)
            except TypeError:
                pred = model.inference(images)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM with {len(images)} frames. Reduce --batch-limit or "
            "--process-res, or let batched mode handle it (set --batch-limit "
            "lower than total frames)."
        ) from None

    depth = np.asarray(pred.depth)
    conf = np.asarray(pred.conf)
    w2c = np.asarray(pred.extrinsics)
    K = np.asarray(pred.intrinsics)
    imgs = np.asarray(pred.processed_images)
    return depth, conf, w2c, K, imgs


def _infer_batched(model, images, chunk_size, use_ray_pose, process_res):
    """Process frames in overlapping chunks, aligning each to a shared world frame.

    Strategy:
      - Chunk the frames with 50% overlap
      - Run DA3 on each chunk independently
      - Align chunk N to chunk N-1 using the overlapping frames' 3D points
        (Procrustes / Sim(3) alignment on the overlap region)
      - Concatenate aligned results
    """
    import torch

    overlap = chunk_size // 2  # 50% overlap
    step = chunk_size - overlap

    # Build chunk boundaries
    chunks = []
    i = 0
    while i < len(images):
        end = min(i + chunk_size, len(images))
        chunks.append((i, end))
        if end == len(images):
            break
        i += step
    log.info("  %d chunks: %s", len(chunks), [(s, e) for s, e in chunks])

    # Process each chunk
    all_depth, all_conf, all_w2c, all_K, all_imgs = [], [], [], [], []
    prev_pred = None  # predictions from previous chunk (for alignment)

    for ci, (start, end) in enumerate(chunks):
        chunk_imgs = images[start:end]
        log.info("  Chunk %d/%d: frames %d-%d (%d frames)",
                 ci + 1, len(chunks), start, end - 1, len(chunk_imgs))

        torch.cuda.empty_cache()
        with torch.inference_mode():
            try:
                pred = model.inference(chunk_imgs, use_ray_pose=use_ray_pose,
                                       process_res=process_res)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM on chunk of {len(chunk_imgs)} frames. Reduce --batch-limit."
                ) from None

        depth = np.asarray(pred.depth)
        conf = np.asarray(pred.conf)
        w2c = np.asarray(pred.extrinsics)
        K = np.asarray(pred.intrinsics)
        imgs = np.asarray(pred.processed_images)

        if ci == 0:
            # First chunk: use as-is, it defines the world frame
            all_depth.append(depth)
            all_conf.append(conf)
            all_w2c.append(w2c)
            all_K.append(K)
            all_imgs.append(imgs)
            prev_depth, prev_conf, prev_w2c, prev_K = depth, conf, w2c, K
        else:
            # Align this chunk to previous using overlap region
            # Overlap: last `overlap` frames of prev == first `overlap` frames of this
            n_overlap = min(overlap, chunks[ci - 1][1] - start)

            # Get 3D points from overlap frames in both coordinate systems
            prev_overlap_w2c = prev_w2c[-n_overlap:]
            prev_overlap_K = prev_K[-n_overlap:]
            prev_overlap_depth = prev_depth[-n_overlap:]

            curr_overlap_w2c = w2c[:n_overlap]
            curr_overlap_K = K[:n_overlap]
            curr_overlap_depth = depth[:n_overlap]

            # Compute Sim(3) transform from current chunk to world frame
            s, R, t = _compute_sim3_alignment(
                prev_overlap_depth, prev_overlap_w2c, prev_overlap_K,
                curr_overlap_depth, curr_overlap_w2c, curr_overlap_K,
            )

            # Apply transform to current chunk's extrinsics
            w2c_aligned = _apply_sim3_to_extrinsics(w2c, s, R, t)
            depth_aligned = depth * s  # scale depths

            # Append only the non-overlapping portion
            all_depth.append(depth_aligned[n_overlap:])
            all_conf.append(conf[n_overlap:])
            all_w2c.append(w2c_aligned[n_overlap:])
            all_K.append(K[n_overlap:])
            all_imgs.append(imgs[n_overlap:])

            prev_depth = depth_aligned
            prev_conf = conf
            prev_w2c = w2c_aligned
            prev_K = K

        del pred
        torch.cuda.empty_cache()

    return (
        np.concatenate(all_depth),
        np.concatenate(all_conf),
        np.concatenate(all_w2c),
        np.concatenate(all_K),
        np.concatenate(all_imgs),
    )


def _compute_sim3_alignment(
    depth_a, w2c_a, K_a,
    depth_b, w2c_b, K_b,
    n_sample: int = 5000,
):
    """Compute Sim(3) transform (s, R, t) that maps points in B's frame to A's frame.

    Correspondence is established *by pixel*: the overlap frames are literally the
    same images in both chunks, so pixel (i,u,v) in A and in B are the same
    physical point. That only holds if both sides are masked identically - hence
    the JOINT valid mask below. Masking each side independently silently shifts
    the correspondence (or crashes on a length mismatch) as soon as either chunk
    emits a single zero/NaN depth.
    """
    pts_a, valid_a = _backproject_frames(depth_a, w2c_a, K_a)  # [P,3], [P] bool
    pts_b, valid_b = _backproject_frames(depth_b, w2c_b, K_b)

    if pts_a.shape != pts_b.shape:
        raise ValueError(
            f"Overlap frames disagree in shape ({pts_a.shape} vs {pts_b.shape}). "
            "Both chunks must be run at the same --process-res."
        )

    joint = valid_a & valid_b
    n_valid = int(joint.sum())
    if n_valid < 100:
        raise RuntimeError(
            f"Only {n_valid} jointly-valid pixels in the chunk overlap - cannot "
            "align. Increase --batch-limit (bigger overlap) or --target-frames."
        )

    idx = np.flatnonzero(joint)
    if len(idx) > n_sample:
        idx = np.random.default_rng(42).choice(idx, n_sample, replace=False)
    pts_a, pts_b = pts_a[idx], pts_b[idx]

    # Solve Sim(3): pts_a = s * R @ pts_b + t
    s, R, t = _umeyama(pts_b, pts_a)
    log.info("    Sim3 alignment: scale=%.4f (from %d/%d valid overlap pixels)",
             s, len(idx), len(joint))
    if not (0.1 < s < 10.0):
        log.warning(
            "    Sim3 scale %.3f is implausible - chunk overlap may be too small "
            "or textureless. Expect a seam here.", s
        )
    return s, R, t


def _backproject_frames(depth, w2c, K):
    """Back-project depth frames to world-space points, keeping pixel order.

    Returns (points [N*H*W, 3], valid [N*H*W] bool). Invalid pixels are KEPT as
    placeholder rows so the array stays index-aligned with the pixel grid -
    the caller intersects masks across chunks. Filtering here would destroy the
    positional correspondence the Sim(3) solve depends on.
    """
    N, H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # [3, H*W]

    all_pts, all_valid = [], []
    for i in range(N):
        d = depth[i].reshape(-1).astype(np.float64)
        valid = np.isfinite(d) & (d > 0)
        d_safe = np.where(valid, d, 1.0)  # avoid NaN/inf propagating through matmul

        K_inv = np.linalg.inv(K[i])
        pts_cam = (K_inv @ uv1) * d_safe[np.newaxis, :]  # [3, H*W]

        R_c2w = w2c[i, :3, :3].T
        t_c2w = -R_c2w @ w2c[i, :3, 3]
        pts_world = (R_c2w @ pts_cam).T + t_c2w  # [H*W, 3]

        all_pts.append(pts_world)
        all_valid.append(valid)

    return np.concatenate(all_pts, axis=0), np.concatenate(all_valid, axis=0)


def _umeyama(src, dst):
    """Umeyama alignment: find s, R, t such that dst ≈ s*R@src + t.

    Returns (s, R, t) where s is scalar, R is [3,3], t is [3,].
    """
    n, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = np.sum(src_c ** 2) / n

    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.eye(dim)
    if d < 0:
        D[-1, -1] = -1

    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / var_src
    t = dst_mean - s * (R @ src_mean)

    return s, R, t


def _apply_sim3_to_extrinsics(w2c, s, R, t):
    """Apply Sim(3) to extrinsics: transform from chunk-local to world frame."""
    N = w2c.shape[0]
    result = np.zeros_like(w2c)
    # w2c_new = w2c_old @ inv(Sim3_world_from_local)
    # For each camera: c2w_old -> apply sim3 -> c2w_new
    for i in range(N):
        R_w2c = w2c[i, :3, :3]
        t_w2c = w2c[i, :3, 3]
        # c2w in local frame
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        # Transform to world frame
        t_c2w_world = s * (R @ t_c2w) + t
        R_c2w_world = R @ R_c2w
        # Back to w2c
        result[i, :3, :3] = R_c2w_world.T
        result[i, :3, 3] = -R_c2w_world.T @ t_c2w_world
    return result


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def fuse_point_cloud(
    imgs, depth, conf, w2c, K, out_dir: Path,
    conf_percentile: float, voxel_size: float, max_depth: float,
) -> Path:
    """Back-project every confident pixel to world space and fuse."""
    import open3d as o3d

    n, h, w = depth.shape
    thr = np.percentile(conf, conf_percentile)
    log.info("Fusing %d depth maps (conf > %.3f, depth < %.1fm)...", n, thr, max_depth)

    all_pts, all_cols = [], []
    us, vs = np.meshgrid(np.arange(w), np.arange(h))

    for i in range(n):
        d = depth[i]
        mask = (conf[i] > thr) & (d > 0.05) & (d < max_depth)
        if mask.sum() == 0:
            continue

        fx, fy = K[i][0, 0], K[i][1, 1]
        cx, cy = K[i][0, 2], K[i][1, 2]

        z = d[mask]
        x = (us[mask] - cx) * z / fx
        y = (vs[mask] - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)                      # camera frame

        R, t = w2c[i][:, :3], w2c[i][:, 3]                          # world->cam
        pts_world = (pts_cam - t) @ R                               # R^T (p - t)

        all_pts.append(pts_world.astype(np.float32))
        all_cols.append((imgs[i][mask] / 255.0).astype(np.float32))

    pts = np.concatenate(all_pts)
    cols = np.concatenate(all_cols)
    log.info("Raw fused points: %.1fM", len(pts) / 1e6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
    )

    out = out_dir / "fused_points.ply"
    o3d.io.write_point_cloud(str(out), pcd)
    log.info("Cleaned cloud: %d points -> %s", len(pcd.points), out)
    return out


# --------------------------------------------------------------------------- #
# COLMAP export (feeds gsplat / Gaussian Splatting / nerfstudio)
# --------------------------------------------------------------------------- #
def export_colmap(imgs, w2c, K, frame_paths: list[Path], out_dir: Path) -> Path:
    """Write a COLMAP text model with DA3 poses (PINHOLE, per-image camera)."""
    import cv2

    sparse = out_dir / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / "colmap" / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    n, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2]

    with open(sparse / "cameras.txt", "w") as fc, open(sparse / "images.txt", "w") as fi:
        fc.write("# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        fi.write("# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i in range(n):
            fx, fy = K[i][0, 0], K[i][1, 1]
            cx, cy = K[i][0, 2], K[i][1, 2]
            fc.write(f"{i+1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

            q = _rotmat_to_qvec(w2c[i][:, :3])
            t = w2c[i][:, 3]
            name = f"frame_{i:06d}.jpg"
            fi.write(
                f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i+1} {name}\n\n"
            )
            cv2.imwrite(str(img_out / name), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))

    (sparse / "points3D.txt").write_text("# empty - dense init comes from PLY\n")
    log.info("COLMAP model exported -> %s", sparse.parent.parent)
    return out_dir / "colmap"


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z), COLMAP convention."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    Kmat = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(Kmat)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if q[0] < 0:
        q = -q
    return q
