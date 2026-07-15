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

log = logging.getLogger("home3d.da3")

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"  # metric scale, CC BY-NC
APACHE_MODEL = "depth-anything/DA3-BASE"                     # permissive, lower accuracy


def run_da3(
    frame_paths: list[Path],
    workdir: Path,
    model_dir: str = DEFAULT_MODEL,
    conf_percentile: float = 40.0,
    voxel_size: float = 0.01,
    max_depth: float = 12.0,
    use_ray_pose: bool = True,
    batch_limit: int = 160,
) -> dict:
    """Run DA3 over keyframes and fuse a world-space metric point cloud."""
    import torch
    from depth_anything_3.api import DepthAnything3

    recon_dir = workdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    if len(frame_paths) > batch_limit:
        log.warning(
            "%d frames > batch_limit=%d. Truncating uniformly. For full-house "
            "long videos use DA3-Streaming (see README) or raise the limit if "
            "you have >24GB VRAM.",
            len(frame_paths), batch_limit,
        )
        idx = np.linspace(0, len(frame_paths) - 1, batch_limit).astype(int)
        frame_paths = [frame_paths[i] for i in idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("No CUDA GPU found - DA3 on CPU will be extremely slow.")

    log.info("Loading DA3 model: %s", model_dir)
    model = DepthAnything3.from_pretrained(model_dir).to(device=device)

    images = [str(p) for p in frame_paths]
    log.info("Running DA3 inference on %d frames (ray pose=%s)...", len(images), use_ray_pose)
    try:
        pred = model.inference(images, use_ray_pose=use_ray_pose)
    except TypeError:
        # older/newer API without the kwarg
        pred = model.inference(images)

    depth = np.asarray(pred.depth)              # [N,H,W] float32, meters (nested model)
    conf = np.asarray(pred.conf)                # [N,H,W]
    w2c = np.asarray(pred.extrinsics)           # [N,3,4] opencv/colmap convention
    K = np.asarray(pred.intrinsics)             # [N,3,3]
    imgs = np.asarray(pred.processed_images)    # [N,H,W,3] uint8

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
