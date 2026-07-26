"""
Stage 3: Point cloud -> CAD-ready assets.

Because the cloud is at METRIC scale (DA3 nested model), everything below is
in real meters:

  1. Gravity alignment      floor normal -> +Z (RANSAC on dominant plane)
  2. Plane segmentation     iterative RANSAC: floor / ceiling / walls,
                            each with fitted plane equation + extent -> planes.json
  3. Watertight-ish mesh    Poisson reconstruction, density-trimmed -> mesh.obj/.stl
  4. Floor plan             horizontal slice at cut height -> wall contours
                            -> polylines in a DXF (opens in AutoCAD/FreeCAD/LibreCAD)

"rough CAD" workflow: import house_plan.dxf for the 2D plan, planes.json for
wall dimensions, mesh.stl/obj for 3D massing in FreeCAD or Blender.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("video2cad.cad")


def extract_cad(
    pcd_path: Path,
    out_dir: Path,
    plane_dist_thresh: float = 0.02,
    min_plane_points: int = 4000,
    max_planes: int = 40,
    slice_height: float = 1.2,
    slice_thickness: float = 0.10,
    grid_res: float = 0.02,
    rescale_height: float | None = None,
) -> dict:
    import open3d as o3d

    out_dir.mkdir(parents=True, exist_ok=True)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    log.info("Loaded %d points", len(pcd.points))

    # 1. gravity alignment ---------------------------------------------------
    # The cloud may still be in arbitrary units here (non-metric checkpoints), so
    # a fixed metric threshold would be meaningless. Use one relative to the
    # cloud's own size, which is scale-invariant.
    diag = float(np.linalg.norm(
        pcd.get_axis_aligned_bounding_box().get_extent()
    ))
    pcd, T_align = _align_to_gravity(pcd, max(1e-6, 0.003 * diag))

    # 1b. rescale if a known height is provided (non-metric models) -----------
    # After this block the cloud is in TRUE METERS, so the metric thresholds
    # (plane_dist_thresh, grid_res, slice_*) apply as-is and must NOT be scaled.
    scale = 1.0
    if rescale_height is not None:
        pts = np.asarray(pcd.points).copy()
        raw_height = np.percentile(pts[:, 2], 98) - np.percentile(pts[:, 2], 2)
        if raw_height > 1e-6:
            scale = rescale_height / raw_height
            log.info(
                "Rescaling: raw height=%.4f units -> %.3f m (scale=%.4f)",
                raw_height, rescale_height, scale,
            )
            pts *= scale
            pts[:, 2] -= np.percentile(pts[:, 2], 2)  # re-zero the floor
            pcd.points = o3d.utility.Vector3dVector(pts)
        else:
            log.warning(
                "Cannot rescale: measured height=%.6f units is degenerate. "
                "The cloud is probably too sparse or has no vertical extent.",
                raw_height,
            )

    np.savetxt(out_dir / "alignment_T.txt", T_align)
    aligned_path = out_dir / "aligned_points.ply"
    o3d.io.write_point_cloud(str(aligned_path), pcd)

    # 2. plane segmentation --------------------------------------------------
    planes = _segment_planes(pcd, plane_dist_thresh, min_plane_points, max_planes, out_dir)
    with open(out_dir / "planes.json", "w") as f:
        json.dump(planes, f, indent=2)
    log.info(
        "Planes: %d walls, %d floors, %d ceilings",
        sum(p["label"] == "wall" for p in planes),
        sum(p["label"] == "floor" for p in planes),
        sum(p["label"] == "ceiling" for p in planes),
    )

    # 3. mesh ------------------------------------------------------------------
    mesh_paths = _poisson_mesh(pcd, out_dir)

    # 4. DXF floor plan ---------------------------------------------------------
    floor_z = min(
        (p["centroid"][2] for p in planes if p["label"] == "floor"),
        default=float(np.percentile(np.asarray(pcd.points)[:, 2], 1)),
    )
    dxf_path = _floor_plan_dxf(
        np.asarray(pcd.points), floor_z + slice_height, slice_thickness, grid_res, out_dir
    )

    return {
        "aligned_cloud": aligned_path,
        "planes": out_dir / "planes.json",
        **mesh_paths,
        "floor_plan_dxf": dxf_path,
    }


# --------------------------------------------------------------------------- #
def _align_to_gravity(pcd, dist):
    """Rotate so the dominant (floor) plane normal is +Z; floor at z=0."""
    import open3d as o3d

    model, inliers = pcd.segment_plane(dist, ransac_n=3, num_iterations=2000)
    n = np.array(model[:3])
    n = n if n[2] >= 0 else -n
    z = np.array([0.0, 0.0, 1.0])

    v = np.cross(n, z)
    c = float(np.dot(n, z))
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    T = np.eye(4)
    T[:3, :3] = R
    pcd = pcd.rotate(R, center=(0, 0, 0))

    # dominant horizontal plane may be the CEILING: if most other points sit
    # below it, flip the scene 180 deg about X so up is truly +Z
    zs = np.asarray(pcd.points)[:, 2]
    plane_z = float(np.median(zs[np.asarray(inliers)]))
    if np.median(np.delete(zs, inliers)) < plane_z:
        F = np.diag([1.0, -1.0, -1.0])
        pcd = pcd.rotate(F, center=(0, 0, 0))
        T[:3, :3] = F @ T[:3, :3]
        log.info("Dominant plane was the ceiling - flipped scene upright")

    # anchor the floor at z=0 using the lowest dense surface
    floor_z = float(np.percentile(np.asarray(pcd.points)[:, 2], 1))
    pcd = pcd.translate((0, 0, -floor_z))
    T[2, 3] = -floor_z
    log.info("Gravity-aligned (floor -> z=0)")
    return pcd, T


def _segment_planes(pcd, dist, min_pts, max_planes, out_dir):
    """Iterative RANSAC plane extraction + semantic labeling."""
    import open3d as o3d

    rest = pcd
    planes, colored = [], []
    palette = np.random.default_rng(0).uniform(0.15, 0.95, size=(max_planes, 3))
    zs = np.asarray(pcd.points)[:, 2]
    z_mid = 0.5 * (np.percentile(zs, 1) + np.percentile(zs, 99))

    for k in range(max_planes):
        if len(rest.points) < min_pts:
            break
        model, inliers = rest.segment_plane(dist, ransac_n=3, num_iterations=1500)
        if len(inliers) < min_pts:
            break

        seg = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)

        n = np.array(model[:3])
        n /= np.linalg.norm(n)
        pts = np.asarray(seg.points)
        centroid = pts.mean(axis=0)
        vert = abs(n[2])

        if vert > 0.9:
            label = "floor" if centroid[2] < z_mid else "ceiling"
        elif vert < 0.2:
            label = "wall"
        else:
            label = "slanted"

        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        planes.append({
            "id": k,
            "label": label,
            "normal": n.round(4).tolist(),
            "d": round(float(model[3]), 4),
            "centroid": centroid.round(3).tolist(),
            "bbox_min": mins.round(3).tolist(),
            "bbox_max": maxs.round(3).tolist(),
            "extent_m": (maxs - mins).round(3).tolist(),
            "n_points": len(inliers),
        })
        seg.paint_uniform_color(palette[k])
        colored.append(seg)

    if colored:
        merged = colored[0]
        for s in colored[1:]:
            merged += s
        o3d.io.write_point_cloud(str(out_dir / "planes_colored.ply"), merged)
    return planes


def _poisson_mesh(pcd, out_dir):
    import open3d as o3d

    log.info("Poisson meshing (depth=10)...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.03))  # trim halo
    mesh = mesh.simplify_quadric_decimation(600_000)
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()

    obj = out_dir / "mesh.obj"
    stl = out_dir / "mesh.stl"
    o3d.io.write_triangle_mesh(str(obj), mesh)
    o3d.io.write_triangle_mesh(str(stl), mesh)
    log.info("Mesh: %d tris -> mesh.obj / mesh.stl", len(mesh.triangles))
    return {"mesh_obj": obj, "mesh_stl": stl}


def _floor_plan_dxf(pts, cut_z, thickness, grid_res, out_dir):
    """Slice the cloud at cut height, rasterize, trace wall contours -> DXF."""
    import cv2
    import ezdxf

    slab = pts[np.abs(pts[:, 2] - cut_z) < thickness / 2]
    if len(slab) < 500:
        log.warning("Floor-plan slice nearly empty at z=%.2fm - skipping DXF", cut_z)
        return None

    xy = slab[:, :2]
    mins = xy.min(axis=0)
    grid = np.zeros(
        (np.ceil((xy.max(axis=0) - mins) / grid_res).astype(int) + 1)[::-1], np.uint8
    )
    ij = ((xy - mins) / grid_res).astype(int)
    grid[ij[:, 1], ij[:, 0]] = 255

    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()

    kept = 0
    for c in contours:
        if cv2.contourArea(c) < 25:  # noise blobs
            continue
        approx = cv2.approxPolyDP(c, epsilon=2.5, closed=True)  # straighten walls
        poly = approx.reshape(-1, 2) * grid_res + mins
        msp.add_lwpolyline(poly.tolist(), close=True, dxfattribs={"layer": "WALLS"})
        kept += 1

    path = out_dir / "house_plan.dxf"
    doc.saveas(path)
    log.info("Floor plan: %d wall polylines (slice @ z=%.2fm) -> %s", kept, cut_z, path)
    return path
