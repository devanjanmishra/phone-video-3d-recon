"""
compare.py — Compare quality of multiple video2cad runs side by side.

Usage:
    python compare.py out_full out_full_hq out_batched
    python compare.py out_*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_run(workdir: Path) -> dict:
    """Load key metrics from a pipeline run."""
    info = {"workdir": str(workdir)}

    # Manifest
    manifest_path = workdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        info["n_frames"] = manifest.get("recon", {}).get("n_frames", "?")
    else:
        info["n_frames"] = "?"

    # Point cloud
    for ply_name in ["recon/fused_points.ply", "cad/aligned_points.ply"]:
        ply_path = workdir / ply_name
        if ply_path.exists():
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(ply_path))
            info["total_points"] = len(pcd.points)
            pts = np.asarray(pcd.points)
            info["bbox_extent"] = (pts.max(0) - pts.min(0)).tolist()
            break
    else:
        info["total_points"] = 0
        info["bbox_extent"] = [0, 0, 0]

    # Planes
    planes_path = workdir / "cad" / "planes.json"
    if planes_path.exists():
        planes = json.loads(planes_path.read_text())
        info["n_planes"] = len(planes)
        info["n_walls"] = sum(1 for p in planes if p["label"] == "wall")
        info["n_floors"] = sum(1 for p in planes if p["label"] == "floor")
        info["n_ceilings"] = sum(1 for p in planes if p["label"] == "ceiling")
        info["plane_points"] = sum(p["n_points"] for p in planes)

        # Wall heights (z-extent of wall planes)
        wall_heights = [p["extent_m"][2] for p in planes if p["label"] == "wall"]
        info["wall_height_mean"] = float(np.mean(wall_heights)) if wall_heights else 0
        info["wall_height_std"] = float(np.std(wall_heights)) if wall_heights else 0

        # Largest wall area (width × height)
        wall_areas = []
        for p in planes:
            if p["label"] == "wall":
                ext = p["extent_m"]
                # Wall: one thin dimension, area = max(ext) * ext[2]
                dims = sorted(ext, reverse=True)
                wall_areas.append(dims[0] * dims[1])
        info["largest_wall_area"] = max(wall_areas) if wall_areas else 0

        # Floor extent
        floor_extents = [p["extent_m"] for p in planes if p["label"] == "floor"]
        if floor_extents:
            biggest_floor = max(floor_extents, key=lambda e: e[0] * e[1])
            info["floor_extent"] = biggest_floor[:2]
        else:
            info["floor_extent"] = [0, 0]
    else:
        info["n_planes"] = 0
        info["n_walls"] = 0
        info["n_floors"] = 0
        info["n_ceilings"] = 0
        info["plane_points"] = 0
        info["wall_height_mean"] = 0
        info["wall_height_std"] = 0
        info["largest_wall_area"] = 0
        info["floor_extent"] = [0, 0]

    # DXF
    info["has_dxf"] = (workdir / "cad" / "house_plan.dxf").exists()

    # Planes colored coverage
    planes_ply = workdir / "cad" / "planes_colored.ply"
    if planes_ply.exists():
        import open3d as o3d
        pcd_c = o3d.io.read_point_cloud(str(planes_ply))
        info["colored_points"] = len(pcd_c.points)
        if pcd_c.has_colors():
            info["n_colors"] = len(np.unique(np.asarray(pcd_c.colors), axis=0))
        else:
            info["n_colors"] = 0
    else:
        info["colored_points"] = 0
        info["n_colors"] = 0

    # Coverage ratio: how much of the cloud is explained by planes
    if info["total_points"] > 0:
        info["plane_coverage"] = info["plane_points"] / info["total_points"]
    else:
        info["plane_coverage"] = 0

    return info


def print_comparison(runs: list[dict]):
    """Print a comparison table."""
    # Header
    headers = [Path(r["workdir"]).name for r in runs]
    col_w = max(16, max(len(h) for h in headers) + 2)

    def row(label, key, fmt=None):
        parts = [f"  {label:24s}"]
        for r in runs:
            val = r.get(key, "—")
            if fmt and val != "—":
                val = fmt(val)
            parts.append(f"{str(val):>{col_w}}")
        print("".join(parts))

    print()
    print("=" * (26 + col_w * len(runs)))
    print("  QUALITY COMPARISON")
    print("=" * (26 + col_w * len(runs)))
    print()
    print(f"  {'Metric':24s}" + "".join(f"{h:>{col_w}}" for h in headers))
    print(f"  {'-'*24}" + "".join(f"  {'-'*(col_w-2)}" for _ in headers))

    row("Frames processed", "n_frames")
    row("Total points", "total_points", lambda v: f"{v:,}")
    row("Points in planes", "plane_points", lambda v: f"{v:,}")
    row("Plane coverage", "plane_coverage", lambda v: f"{v:.0%}")
    row("Total planes", "n_planes")
    row("Walls", "n_walls")
    row("Floors", "n_floors")
    row("Ceilings", "n_ceilings")
    row("Wall height (mean)", "wall_height_mean", lambda v: f"{v:.2f}m")
    row("Wall height (std)", "wall_height_std", lambda v: f"{v:.3f}m")
    row("Largest wall area", "largest_wall_area", lambda v: f"{v:.1f}m²")
    row("Floor extent", "floor_extent", lambda v: f"{v[0]:.1f}×{v[1]:.1f}m")
    row("Has DXF", "has_dxf", lambda v: "✓" if v else "✗")
    row("Bbox extent", "bbox_extent",
        lambda v: f"{v[0]:.1f}×{v[1]:.1f}×{v[2]:.1f}m")

    print()

    # Score (simple heuristic)
    print(f"  {'Quality score':24s}", end="")
    for r in runs:
        # Heuristic, weights chosen to total exactly 100:
        #   walls 30 | coverage 25 | DXF 15 | density 15 | height consistency 10 | floor 5
        score = (
            min(r["n_walls"] / 10, 1.0) * 30 +
            min(r["plane_coverage"], 1.0) * 25 +
            (15 if r["has_dxf"] else 0) +
            min(r["total_points"] / 200_000, 1.0) * 15 +
            (10 if 0 < r["wall_height_std"] < 0.2 else 0) +
            min(r["n_floors"], 1) * 5
        )
        print(f"{score:>{col_w}.0f}", end="")
    print(" / 100")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare.py <workdir1> <workdir2> [workdir3] ...")
        print("       python compare.py out_*")
        sys.exit(1)

    workdirs = [Path(p) for p in sys.argv[1:] if Path(p).is_dir()]
    if not workdirs:
        print("No valid workdirs found.")
        sys.exit(1)

    runs = []
    for wd in sorted(workdirs):
        if (wd / "manifest.json").exists() or (wd / "cad" / "planes.json").exists():
            runs.append(load_run(wd))

    if not runs:
        print("No pipeline outputs found in the given directories.")
        sys.exit(1)

    print_comparison(runs)


if __name__ == "__main__":
    main()
