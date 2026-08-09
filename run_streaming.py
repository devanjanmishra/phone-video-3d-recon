"""Convenience command for DA3-Streaming reconstruction."""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-video reconstruction with DA3-Streaming")
    parser.add_argument("video", help="input walkthrough video")
    parser.add_argument("workdir", help="output directory")
    parser.add_argument("--target-frames", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="frames per chunk; 10 is the safe starting point for 16 GB GPUs")
    parser.add_argument("--process-res", type=int, default=336,
                        help="DA3 processing resolution; 336 is the safe starting point for 16 GB GPUs")
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--streaming-dir")
    parser.add_argument("--stream-config")
    parser.add_argument("--rescale-height", type=float)
    parser.add_argument("--slice-height", type=float, default=1.2)
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "video2cad.cli", "--video", args.video,
        "--workdir", args.workdir, "--stages", "frames,stream,cad,viz",
        "--target-frames", str(args.target_frames), "--chunk-size", str(args.chunk_size),
        "--process-res", str(args.process_res),
        "--slice-height", str(args.slice_height),
    ]
    if args.overlap is not None:
        command.extend(("--overlap", str(args.overlap)))
    if args.no_loop:
        command.append("--no-loop")
    if args.streaming_dir:
        command.extend(("--streaming-dir", args.streaming_dir))
    if args.stream_config:
        command.extend(("--stream-config", args.stream_config))
    if args.rescale_height is not None:
        command.extend(("--rescale-height", str(args.rescale_height)))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()