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
    ap.add_argument("--target-frames", type=int, default=None,
                    help="Keyframes to extract (default: as many as fit in one "
                         "pass - 16 at 504px, 40 at 336px)")
    ap.add_argument("--low-res", action="store_true",
                    help="Use 336px resolution (fits more frames, lower detail)")
    ap.add_argument("--process-res", type=int, default=None,
                    help="Override DA3 processing resolution")
    ap.add_argument("--rescale-height", type=float, default=None,
                    help="Known floor-to-ceiling or door height (m) for non-metric models")

    args = ap.parse_args()

    # Resolution presets: `vram_ceiling` is the frame count that fits in one
    # forward pass on a 16 GB fp32 GPU at this resolution.
    if args.low_res:
        process_res, vram_ceiling = 336, 40
    else:
        process_res, vram_ceiling = 504, 16
    if args.process_res:
        process_res = args.process_res
    if args.target_frames is None:
        args.target_frames = vram_ceiling

    # Single-pass means EVERY frame in ONE forward pass. batch_limit must
    # therefore be >= target_frames, or video2cad silently switches to chunked
    # mode and this script stops doing what its name says.
    batch_limit = args.target_frames
    if args.target_frames > vram_ceiling:
        print(f"  ! {args.target_frames} frames at {process_res}px exceeds the "
              f"~{vram_ceiling}-frame ceiling for a 16 GB fp32 GPU.")
        print(f"  ! Single-pass will still be attempted and may OOM. For more "
              f"frames use: python run_batch.py {args.video} {args.workdir}")
        print()

    # Build command
    cmd = [
        sys.executable, "-m", "video2cad.cli",
        "--video", args.video,
        "--workdir", args.workdir,
        "--model", args.model,
        "--target-frames", str(args.target_frames),
        "--batch-limit", str(batch_limit),
        "--process-res", str(process_res),
    ]
    if args.rescale_height is not None:
        cmd += ["--rescale-height", str(args.rescale_height)]

    print(f"=== Single-pass mode ===")
    print(f"  Model:         {args.model}")
    print(f"  Target frames: {args.target_frames} (all in one pass; "
          f"~{vram_ceiling} fits 16 GB fp32)")
    print(f"  Resolution:    {process_res}px")
    print(f"  Rescale:       {args.rescale_height or 'none'}")
    print(f"  Command:       {' '.join(cmd)}")
    print()

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
