"""
Stage 1: Video -> keyframes.

Smart extraction for reconstruction quality:
  - uniform temporal sampling down to a target frame budget
  - per-window sharpness ranking (variance of Laplacian) so we keep the
    least motion-blurred frame in each window instead of a blind stride
  - optional resize of the long edge (DA3 works at ~504-1008 px well)

Pure OpenCV, no ffmpeg dependency (works out of the box on Windows).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("video2cad.frames")


def _sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian - standard blur metric (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    target_frames: int = 200,
    long_edge: int = 1008,
    min_sharpness: float = 40.0,
) -> list[Path]:
    """
    Extract `target_frames` keyframes from a video.

    The video timeline is split into `target_frames` equal windows; inside
    each window every candidate frame is scored for sharpness and the best
    one is kept (if it clears `min_sharpness`).

    Returns list of written frame paths (ordered by time).
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("Video: %d frames @ %.1f fps (%.1fs)", n_total, fps, n_total / fps)

    if n_total <= 0:
        # some containers don't report frame count; fall back to full decode
        n_total = _count_frames(cap)

    target_frames = min(target_frames, n_total)
    # window boundaries (inclusive start, exclusive end)
    bounds = np.linspace(0, n_total, target_frames + 1, dtype=int)

    written: list[Path] = []
    frame_idx = 0
    win = 0
    best_frame, best_score, best_idx = None, -1.0, -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # advance window if needed, flushing the best frame of the old one
        while win < target_frames and frame_idx >= bounds[win + 1]:
            best_frame, best_score, best_idx, path = _flush(
                best_frame, best_score, best_idx, out_dir, long_edge, min_sharpness
            )
            if path:
                written.append(path)
            win += 1

        if win >= target_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # score on a downscaled copy for speed
        small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
        s = _sharpness(small)
        if s > best_score:
            best_frame, best_score, best_idx = frame, s, frame_idx

        frame_idx += 1

    # flush the last window
    _, _, _, path = _flush(
        best_frame, best_score, best_idx, out_dir, long_edge, min_sharpness
    )
    if path:
        written.append(path)

    cap.release()
    log.info("Kept %d / %d requested keyframes -> %s", len(written), target_frames, out_dir)
    if len(written) < target_frames * 0.7:
        log.warning(
            "Many frames rejected for blur. Re-record moving slower, or lower min_sharpness."
        )
    return written


def _flush(best_frame, best_score, best_idx, out_dir: Path, long_edge: int, min_sharpness: float):
    """Write the current window's best frame if sharp enough; reset accumulators."""
    path = None
    if best_frame is not None and best_score >= min_sharpness:
        h, w = best_frame.shape[:2]
        scale = long_edge / max(h, w)
        if scale < 1.0:
            best_frame = cv2.resize(
                best_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        path = out_dir / f"frame_{best_idx:06d}.jpg"
        cv2.imwrite(str(path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return None, -1.0, -1, path


def _count_frames(cap) -> int:
    n = 0
    while cap.read()[0]:
        n += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return n
