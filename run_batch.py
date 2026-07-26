"""
run_batch.py — Batched reconstruction (overlapping chunks, unlimited frames).
Processes frames in windows with Sim(3) alignment between chunks.
No VRAM limit on total frame count.

Usage:
    python run_batch.py home.mp4 out_batch
    python run_batch.py home.mp4 out_batch --high-model
    python run_batch.py home.mp4 out_batch --target-frames 200 --rescale-height 2.1
    python run_batch.py home.mp4 out_batch --high-res

Presets:
    (default)     DA3-SMALL, 336px, 16 frames/chunk — balanced speed/quality
    --high-model  DA3-LARGE, 336px, 8 frames/chunk  — better depth, slower
    --high-res    DA3-SMALL, 504px, 16 frames/chunk — sharper per-frame detail
    --low-res     DA3-SMALL, 336px, 40 frames/chunk — max coverage per chunk
"""

import subprocess
import sys


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Batched reconstruction (overlapping chunks, unlimited frames)")
    ap.add_argument("video", help="Input video file")
    ap.add_argument("workdir", help="Output directory")
    ap.add_argument("--model", default=None,
                    help="DA3 model (overrides preset)")
    ap.add_argument("--target-frames", type=int, default=120,
                    help="Number of keyframes to extract (default: 120)")
    ap.add_argument("--high-model", action="store_true",
                    help="Use DA3-LARGE (better quality, smaller chunks)")
    ap.add_argument("--high-res", action="store_true",
                    help="Use 504px resolution (sharper detail, fewer frames/chunk)")
    ap.add_argument("--low-res", action="store_true",
                    help="Use 336px + 40 frames/chunk (fastest, max coverage)")
    ap.add_argument("--batch-limit", type=int, default=None,
                    help="Override chunk size (frames per forward pass)")
    ap.add_argument("--process-res", type=int, default=None,
                    help="Override DA3 processing resolution")
    ap.add_argument("--rescale-height", type=float, default=None,
                    help="Known floor-to-ceiling or door height (m) for non-metric models")

    args = ap.parse_args()

    # Preset logic
    if args.high_model:
        model = "depth-anything/DA3-LARGE-1.1"
        batch_limit = 8
        process_res = 336
    elif args.high_res:
        model = "depth-anything/DA3-SMALL"
        batch_limit = 16
        process_res = 504
    elif args.low_res:
        model = "depth-anything/DA3-SMALL"
        batch_limit = 40
        process_res = 336
    else:
        # Default: balanced
        model = "depth-anything/DA3-SMALL"
        batch_limit = 16
        process_res = 336

    # Explicit overrides
    if args.model:
        model = args.model
    if args.batch_limit:
        batch_limit = args.batch_limit
    if args.process_res:
        process_res = args.process_res

    n_chunks_approx = max(1, (args.target_frames - 1) // (batch_limit // 2))

    # Build command
    cmd = [
        sys.executable, "-m", "video2cad.cli",
        "--video", args.video,
        "--workdir", args.workdir,
        "--model", model,
        "--target-frames", str(args.target_frames),
        "--batch-limit", str(batch_limit),
        "--process-res", str(process_res),
    ]
    if args.rescale_height is not None:
        cmd += ["--rescale-height", str(args.rescale_height)]

    print(f"=== Batch mode (overlapping chunks) ===")
    print(f"  Model:         {model}")
    print(f"  Target frames: {args.target_frames}")
    print(f"  Chunk size:    {batch_limit} frames (50% overlap)")
    print(f"  Resolution:    {process_res}px")
    print(f"  Est. chunks:   ~{n_chunks_approx}")
    print(f"  Rescale:       {args.rescale_height or 'none'}")
    print(f"  Command:       {' '.join(cmd)}")
    print()

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
