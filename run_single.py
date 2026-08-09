"""
run_single.py — Single-pass reconstruction (all frames in one forward pass).
Best quality per frame but limited by VRAM.

Usage:
    python run_single.py home.mp4 out_single
    python run_single.py home.mp4 out_single --low-res
    python run_single.py home.mp4 out_single --model depth-anything/DA3-BASE
    python run_single.py home.mp4 out_single --rescale-height 2.1
"""

import subprocess
import sys


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Single-pass reconstruction (all frames in one forward pass)")
    ap.add_argument("video", help="Input video file")
    ap.add_argument("workdir", help="Output directory")
    ap.add_argument("--model", default="depth-anything/DA3-SMALL",
                    help="DA3 model (default: DA3-SMALL)")
    ap.add_argument("--target-frames", type=int, default=60,
                    help="Number of keyframes to extract (default: 60)")
    ap.add_argument("--low-res", action="store_true",
                    help="Use 336px resolution (fits more frames, lower detail)")
    ap.add_argument("--rescale-height", type=float, default=None,
                    help="Known floor-to-ceiling or door height (m) for non-metric models")

    args = ap.parse_args()

    # Resolution presets
    if args.low_res:
        process_res = 336
        max_frames = 40
    else:
        process_res = 504
        max_frames = 16

    # Build command
    cmd = [
        sys.executable, "-m", "video2cad.cli",
        "--video", args.video,
        "--workdir", args.workdir,
        "--model", args.model,
        "--target-frames", str(args.target_frames),
        "--max-frames", str(max_frames),
        "--process-res", str(process_res),
    ]
    if args.rescale_height is not None:
        cmd += ["--rescale-height", str(args.rescale_height)]

    print(f"=== Single-pass mode ===")
    print(f"  Model:         {args.model}")
    print(f"  Target frames: {args.target_frames} (max {max_frames} in one pass)")
    print(f"  Resolution:    {process_res}px")
    print(f"  Rescale:       {args.rescale_height or 'none'}")
    print(f"  Command:       {' '.join(cmd)}")
    print()

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
