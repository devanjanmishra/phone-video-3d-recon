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
    max_frames: int = 16,
    device_str: str = "auto",
    dtype_str: str = "auto",
    process_res: int = 504,
) -> dict:
    """Run DA3 over keyframes and fuse a world-space metric point cloud."""
    import torch
    from depth_anything_3.api import DepthAnything3

    recon_dir = workdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    if len(frame_paths) > max_frames:
        raise ValueError(
            f"Single-pass reconstruction accepts at most {max_frames} frames; "
            "use the 'stream' stage for longer captures."
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

    log.info("Running DA3 single-pass inference on %d frames (ray pose=%s, process_res=%d)...",
             len(images), use_ray_pose, process_res)
    depth, conf, w2c, K, imgs = _infer_single(
        model, images, use_ray_pose, process_res)

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
            f"CUDA OOM with {len(images)} frames. Reduce --max-frames or "
            "--process-res, or use the streaming path for longer captures."
        ) from None

    depth = np.asarray(pred.depth)
    conf = np.asarray(pred.conf)
    w2c = np.asarray(pred.extrinsics)
    K = np.asarray(pred.intrinsics)
    imgs = np.asarray(pred.processed_images)
    return depth, conf, w2c, K, imgs


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
