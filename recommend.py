"""
recommend.py — Recommend single-pass or streaming reconstruction settings.

Usage:
    python recommend.py                          # just check GPU
    python recommend.py --video home.mp4         # check GPU + video info
    python recommend.py --vram 8                 # manual VRAM override (GB)

Prints a recommended command to copy-paste.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def get_gpu_info() -> tuple[str, float, int]:
    """Returns (gpu_name, vram_gb, compute_capability_major)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "none", 0.0, 0
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        major, _ = torch.cuda.get_device_capability(0)
        return name, vram, major
    except ImportError:
        return "unknown (torch not installed)", 0.0, 0


def get_video_info(path: str) -> dict:
    """Get video duration, fps, frame count, resolution."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] > 0 else 0
    info["is_portrait"] = info["height"] > info["width"]
    cap.release()
    return info


def recommend(vram_gb: float, cc_major: int, video_info: dict | None) -> dict:
    """Return recommended configuration."""
    # Scale factor: tested on 16GB. For other VRAM, scale proportionally.
    # bf16 (Ampere+) roughly doubles effective capacity.
    effective_vram = vram_gb
    if cc_major >= 8:
        effective_vram *= 2  # bf16 halves memory per parameter

    scale = effective_vram / 16.0

    # Pick model based on VRAM
    if effective_vram >= 48:
        model = "DA3NESTED-GIANT-LARGE-1.1"
        process_res = 504
        max_frames = 80
        metric = True
    elif effective_vram >= 24:
        model = "DA3-LARGE-1.1"
        process_res = 504
        max_frames = int(16 * scale)
        metric = False
    elif effective_vram >= 12:
        model = "DA3-SMALL"
        process_res = 336
        max_frames = int(40 * scale)
        metric = False
    else:
        model = "DA3-SMALL"
        process_res = 336
        max_frames = max(4, int(40 * scale))
        metric = False

    max_frames = max(4, min(max_frames, 160))

    # Target frames based on video duration
    if video_info and video_info.get("duration_s", 0) > 0:
        dur = video_info["duration_s"]
        # ~3 frames per second of video is a good baseline
        target_frames = max(30, min(int(dur * 3), 300))
    else:
        target_frames = 120

    mode = "single" if target_frames <= max_frames else "stream"

    return {
        "model": f"depth-anything/{model}",
        "process_res": process_res,
        "max_frames": max_frames,
        "stream_chunk_size": 10 if cc_major < 8 and vram_gb < 20 else 20,
        "target_frames": target_frames,
        "mode": mode,
        "metric": metric,
        "needs_rescale": not metric,
    }


def main():
    ap = argparse.ArgumentParser(description="Recommend video2cad settings for your hardware")
    ap.add_argument("--video", help="Input video (optional, for duration-based recommendation)")
    ap.add_argument("--vram", type=float, default=None, help="Override VRAM in GB")
    args = ap.parse_args()

    # GPU info
    gpu_name, vram_gb, cc_major = get_gpu_info()
    if args.vram:
        vram_gb = args.vram

    print("=" * 60)
    print("  video2cad — Hardware Recommendation")
    print("=" * 60)
    print()
    print(f"  GPU:           {gpu_name}")
    print(f"  VRAM:          {vram_gb:.1f} GB")
    print(f"  Compute:       sm_{cc_major}x ({'bf16 supported' if cc_major >= 8 else 'fp32 only'})")

    # Video info
    video_info = None
    if args.video:
        video_info = get_video_info(args.video)
        if video_info:
            print()
            print(f"  Video:         {args.video}")
            print(f"  Duration:      {video_info['duration_s']:.1f}s")
            print(f"  Resolution:    {video_info['width']}×{video_info['height']}")
            print(f"  FPS:           {video_info['fps']:.1f}")
            if video_info["is_portrait"]:
                print(f"  ⚠ Portrait orientation — DA3 works best with landscape video")

    if vram_gb == 0:
        print("\n  ❌ No CUDA GPU detected. video2cad requires an NVIDIA GPU.")
        sys.exit(1)

    # Recommendation
    rec = recommend(vram_gb, cc_major, video_info)
    print()
    print("-" * 60)
    print("  RECOMMENDATION")
    print("-" * 60)
    print()
    print(f"  Mode:          {rec['mode']}" +
          (" (frames fit in one chunk)" if rec['mode'] == 'single'
           else " (DA3-Streaming chunks)"))
    print(f"  Model:         {rec['model']}")
    print(f"  Resolution:    {rec['process_res']}px")
    print(f"  Single-pass limit: {rec['max_frames']} frames")
    print(f"  Target frames: {rec['target_frames']}")
    print(f"  Metric scale:  {'yes' if rec['metric'] else 'no — use --rescale-height'}")
    print()

    # Build command
    script = "run_single.py" if rec["mode"] == "single" else "run_streaming.py"
    video_arg = args.video or "<your_video.mp4>"
    cmd_parts = [
        f"python {script} {video_arg} output",
        f"--target-frames {rec['target_frames']}",
    ]
    if rec["mode"] == "stream":
        cmd_parts.extend((
            f"--chunk-size {rec['stream_chunk_size']}",
            "--process-res 336",
            "--no-loop",
        ))
    else:
        cmd_parts.append(f"--model {rec['model']}")
    if rec["mode"] == "single" and rec["process_res"] != 504:
        cmd_parts.append("--low-res")
    if rec["needs_rescale"]:
        cmd_parts.append("--rescale-height 2.1  # ← set to your door/ceiling height in meters")

    print("  Suggested command:")
    print()
    print(f"    {' '.join(cmd_parts)}")
    print()

    # Also show the direct CLI equivalent.
    cli_parts = ["video2cad", f"--video {video_arg}", "--workdir output"]
    if rec["mode"] == "stream":
        cli_parts.extend((
            "--stages frames,stream,cad,viz",
            f"--target-frames {rec['target_frames']}",
            f"--chunk-size {rec['stream_chunk_size']}",
            "--process-res 336",
            "--no-loop",
        ))
    else:
        cli_parts.extend((
            f"--model {rec['model']}",
            f"--target-frames {rec['target_frames']}",
            f"--max-frames {rec['max_frames']}",
            f"--process-res {rec['process_res']}",
        ))
    if rec["needs_rescale"]:
        cli_parts.append("--rescale-height 2.1")
    print("  Or directly:")
    print()
    print(f"    {' '.join(cli_parts)}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
