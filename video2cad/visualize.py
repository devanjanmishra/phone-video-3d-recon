"""
Visualization utilities: turntable point-cloud GIF and depth-montage GIF.

Usage (standalone):
    python -m video2cad.visualize --workdir out
    python -m video2cad.visualize --workdir out --mode turntable
    python -m video2cad.visualize --workdir out --mode depth
    python -m video2cad.visualize --workdir out --mode all

Or from the main CLI:
    video2cad --workdir out --stages viz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("video2cad.visualize")


# ---------------------------------------------------------------------------
# A) Turntable point-cloud GIF
# ---------------------------------------------------------------------------

def turntable_gif(
    ply_path: Path,
    out_path: Path,
    n_frames: int = 60,
    width: int = 800,
    height: int = 600,
    point_size: float = 3.0,
    fps: int = 20,
    bg_color: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> Path:
    """Render an orbiting-camera GIF around a .ply point cloud."""
    import open3d as o3d
    from PIL import Image

    log.info("Loading point cloud: %s", ply_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {ply_path}")

    # If no colors, paint uniform teal
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.3, 0.8, 0.7])

    center = pcd.get_center()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    radius = extent * 1.2

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    renderer.scene.set_background(np.array([*bg_color, 1.0]))
    renderer.scene.add_geometry("cloud", pcd, mat)

    frames: list[Image.Image] = []
    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames
        eye = center + radius * np.array([np.cos(angle), np.sin(angle), 0.4])

        renderer.setup_camera(
            o3d.camera.PinholeCameraIntrinsic(
                width, height, width * 0.8, width * 0.8, width / 2, height / 2
            ),
            np.eye(4),
        )
        renderer.scene.camera.look_at(center, eye, [0, 0, 1])

        img = np.asarray(renderer.render_to_image())
        frames.append(Image.fromarray(img))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out_path),
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    log.info("Turntable GIF (%d frames) -> %s", len(frames), out_path)
    return out_path


# ---------------------------------------------------------------------------
# D) Depth-map montage GIF
# ---------------------------------------------------------------------------

def _colorize_depth(depth: np.ndarray, max_depth: float | None = None) -> np.ndarray:
    """depth [H,W] float -> [H,W,3] uint8 turbo-colormap image."""
    import matplotlib.cm as cm

    d = depth.copy()
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    if max_depth is None:
        max_depth = float(np.percentile(d[valid], 98))
    d = np.clip(d, 0, max_depth) / max_depth  # 0..1
    d[~valid] = 1.0
    colored = (cm.turbo(d)[:, :, :3] * 255).astype(np.uint8)
    return colored


def depth_montage_gif(
    npz_path: Path,
    frames_dir: Path,
    out_path: Path,
    max_panels: int = 30,
    panel_width: int = 400,
    fps: int = 4,
    max_depth: float | None = None,
) -> Path:
    """Create a GIF scrolling through frame + depth side-by-side panels."""
    import cv2
    from PIL import Image

    log.info("Loading reconstruction data: %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    depths = data["depth"]          # [N,H,W]
    image_paths = data["images"]    # [N] strings — original frame paths used

    n = len(depths)
    step = max(1, n // max_panels)
    indices = list(range(0, n, step))[:max_panels]

    panels: list[Image.Image] = []
    for idx in indices:
        # Load the actual frame image used by DA3
        img_path = str(image_paths[idx])
        frame = cv2.imread(img_path)
        if frame is None:
            log.warning("Cannot read frame %s, skipping", img_path)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth_colored = _colorize_depth(depths[idx], max_depth=max_depth)

        # Resize both to panel_width
        h, w = frame.shape[:2]
        scale = panel_width / w
        new_h = int(h * scale)
        frame_r = cv2.resize(frame, (panel_width, new_h))
        depth_r = cv2.resize(depth_colored, (panel_width, new_h))

        # Side-by-side
        combined = np.concatenate([frame_r, depth_r], axis=1)

        # Add frame index label
        cv2.putText(
            combined,
            f"Frame {idx}/{n}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        panels.append(Image.fromarray(combined))

    if not panels:
        raise ValueError("No panels could be generated — check frame paths in recon.npz")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panels[0].save(
        str(out_path),
        save_all=True,
        append_images=panels[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    log.info("Depth montage GIF (%d panels) -> %s", len(panels), out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def run_visualize(workdir: Path, mode: str = "all") -> dict[str, Path]:
    """Generate visualization GIFs from pipeline outputs."""
    results: dict[str, Path] = {}

    if mode in ("turntable", "all"):
        # Prefer planes_colored if it exists, else aligned_points, else fused
        for candidate in [
            workdir / "cad" / "planes_colored.ply",
            workdir / "cad" / "aligned_points.ply",
            workdir / "recon" / "fused_points.ply",
        ]:
            if candidate.exists():
                ply = candidate
                break
        else:
            log.warning("No point cloud found for turntable GIF")
            ply = None

        if ply is not None:
            out = workdir / "viz" / "turntable.gif"
            try:
                results["turntable"] = turntable_gif(ply, out)
            except Exception as e:
                log.error("Turntable GIF failed: %s", e)

    if mode in ("depth", "all"):
        npz = workdir / "recon" / "recon.npz"
        frames_dir = workdir / "frames"
        if npz.exists():
            out = workdir / "viz" / "depth_montage.gif"
            try:
                results["depth_montage"] = depth_montage_gif(npz, frames_dir, out)
            except Exception as e:
                log.error("Depth montage GIF failed: %s", e)
        else:
            log.warning("No recon.npz found for depth montage")

    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize video2cad outputs as GIFs")
    parser.add_argument("--workdir", type=Path, required=True, help="Pipeline output directory")
    parser.add_argument(
        "--mode",
        choices=["turntable", "depth", "all"],
        default="all",
        help="Which GIF(s) to generate (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s %(message)s")
    results = run_visualize(args.workdir, args.mode)
    for name, path in results.items():
        print(f"  {name:20s} {path}")


if __name__ == "__main__":
    main()
