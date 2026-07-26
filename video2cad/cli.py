"""
video2cad - monocular home-video -> metric point cloud -> rough CAD.

Usage (Windows / anywhere):
    video2cad --video C:\\videos\\home.mp4 --workdir C:\\recon\\myhome
    python -m video2cad.cli --video home.mp4 --workdir out --stages frames,recon,cad
    python run.py --workdir out --stages cad               # re-run CAD only
    video2cad --workdir out --stages viz              # generate GIFs only

Stages:
    frames  video -> sharp keyframes                (CPU, fast)
    recon   DA3 -> poses + metric depth + fused PLY (GPU, minutes)
    cad     planes + mesh + DXF floor plan          (CPU, ~1-2 min)
    viz     turntable + depth-montage GIFs           (CPU, ~30 s)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-14s %(message)s")
log = logging.getLogger("video2cad")


def main() -> int:
    ap = argparse.ArgumentParser(description="Home video -> 3D point cloud -> rough CAD")
    ap.add_argument("--video", type=Path, help="input walkthrough video (mp4/mov/...)")
    ap.add_argument("--workdir", type=Path, required=True, help="output directory")
    ap.add_argument("--stages", default="frames,recon,cad,viz",
                    help="comma list: frames,recon,cad,viz")
    # frames
    ap.add_argument("--target-frames", type=int, default=200)
    ap.add_argument("--long-edge", type=int, default=1008)
    ap.add_argument("--min-sharpness", type=float, default=40.0)
    # recon
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                    help="DA3 model (use depth-anything/DA3-BASE for Apache-2.0 / low VRAM)")
    ap.add_argument("--conf-percentile", type=float, default=40.0,
                    help="drop pixels below this confidence percentile")
    ap.add_argument("--voxel", type=float, default=0.01, help="fusion voxel size (m)")
    ap.add_argument("--max-depth", type=float, default=12.0, help="clip depth beyond (m)")
    ap.add_argument("--batch-limit", type=int, default=160,
                    help="max frames per DA3 forward pass (VRAM bound)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--dtype", default="auto",
                    help="auto | float16 | bfloat16 | float32 (Turing GPUs need float16)")
    ap.add_argument("--process-res", type=int, default=504,
                    help="DA3 internal processing resolution (lower = less VRAM, more frames)")
    # cad
    ap.add_argument("--slice-height", type=float, default=1.2,
                    help="floor-plan cut height above floor (m)")
    ap.add_argument("--rescale-height", type=float, default=None,
                    help="known floor-to-ceiling or door height (m) to rescale non-metric models")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    args.workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.workdir / "frames"
    manifest: dict = {}
    mpath = args.workdir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())

    if "frames" in stages:
        if not args.video or not args.video.exists():
            log.error("Stage 'frames' needs --video pointing to an existing file.")
            return 1
        from video2cad.frames import extract_frames
        paths = extract_frames(
            args.video, frames_dir,
            target_frames=args.target_frames,
            long_edge=args.long_edge,
            min_sharpness=args.min_sharpness,
        )
        manifest["frames"] = [str(p) for p in paths]

    if "recon" in stages:
        frame_paths = [Path(p) for p in manifest.get("frames", [])] or sorted(
            frames_dir.glob("*.jpg")
        )
        if not frame_paths:
            log.error("No frames found - run the 'frames' stage first.")
            return 1
        from video2cad.da3_stage import run_da3
        res = run_da3(
            frame_paths, args.workdir,
            model_dir=args.model,
            conf_percentile=args.conf_percentile,
            voxel_size=args.voxel,
            max_depth=args.max_depth,
            batch_limit=args.batch_limit,
            device_str=args.device,
            dtype_str=args.dtype,
            process_res=args.process_res,
        )
        manifest["recon"] = {k: str(v) for k, v in res.items()}

    if "cad" in stages:
        pcd = Path(manifest.get("recon", {}).get("point_cloud",
                   args.workdir / "recon" / "fused_points.ply"))
        if not pcd.exists():
            log.error("No fused_points.ply - run the 'recon' stage first.")
            return 1
        from video2cad.cad_stage import extract_cad
        res = extract_cad(pcd, args.workdir / "cad", slice_height=args.slice_height,
                          rescale_height=args.rescale_height)
        manifest["cad"] = {k: str(v) for k, v in res.items() if v}

    if "viz" in stages:
        from video2cad.visualize import run_visualize
        viz_res = run_visualize(args.workdir, mode="all")
        manifest["viz"] = {k: str(v) for k, v in viz_res.items()}

    mpath.write_text(json.dumps(manifest, indent=2))
    log.info("Done. Manifest -> %s", mpath)

    if "viz" in manifest:
        print("\n=== Visualizations ===")
        for k, v in manifest["viz"].items():
            print(f"  {k:20s} {v}")

    if "cad" in manifest:
        print("\n=== Deliverables ===")
        for k, v in manifest["cad"].items():
            print(f"  {k:16s} {v}")
        print("  DXF opens in AutoCAD / FreeCAD / LibreCAD; mesh.stl imports into FreeCAD "
              "(Part > Shape from mesh) for solid modeling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
