"""Run DA3-Streaming and normalize its point cloud for the CAD pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("video2cad.streaming")


def _find_streaming_dir(explicit: str | None) -> Path:
    """Locate the upstream DA3-Streaming directory."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if environment_dir := os.environ.get("DA3_STREAMING_DIR"):
        candidates.append(Path(environment_dir))

    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            repo_root / "da3_streaming",
            repo_root / "Depth-Anything-3" / "da3_streaming",
            repo_root / "third_party" / "da3_streaming",
        ]
    )
    for candidate in candidates:
        if (candidate / "da3_streaming.py").is_file():
            return candidate.resolve()

    locations = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "DA3-Streaming was not found. Pass --streaming-dir or set "
        "DA3_STREAMING_DIR. Looked in:\n  " + locations
    )


def run_streaming(
    frame_paths: list[Path],
    workdir: Path,
    streaming_dir: str | None = None,
    config: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    loop_enable: bool | None = None,
    process_res: int | None = None,
) -> dict:
    """Process ordered frames with upstream DA3-Streaming."""
    streaming_path = _find_streaming_dir(streaming_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    recon_dir = workdir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    image_dir = workdir / "streaming_input"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir()
    for index, frame_path in enumerate(frame_paths):
        destination = image_dir / f"frame_{index:06d}{frame_path.suffix.lower()}"
        shutil.copy2(frame_path, destination)

    config_path = _resolve_config(
        streaming_path, config, workdir, chunk_size, overlap, loop_enable, process_res
    )
    output_dir = recon_dir / "streaming"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        sys.executable,
        "da3_streaming.py",
        "--image_dir",
        str(image_dir.resolve()),
        "--config",
        str(config_path.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(streaming_path), environment.get("PYTHONPATH")))
    )
    log.info("Running DA3-Streaming from %s", streaming_path)
    subprocess.run(command, cwd=streaming_path, env=environment, check=True)

    point_cloud = _normalize_outputs(output_dir, recon_dir)
    return {
        "point_cloud": point_cloud,
        "streaming_raw": output_dir,
        "poses": output_dir / "camera_poses.txt",
        "n_frames": len(frame_paths),
    }


def _resolve_config(
    streaming_dir: Path,
    config: str | None,
    workdir: Path,
    chunk_size: int | None,
    overlap: int | None,
    loop_enable: bool | None,
    process_res: int | None,
) -> Path:
    base_config = Path(config).expanduser() if config else streaming_dir / "configs" / "base_config.yaml"
    if not base_config.is_file():
        raise FileNotFoundError(f"Streaming config not found: {base_config}")
    if chunk_size is None and overlap is None and loop_enable is None and process_res is None:
        return base_config

    import yaml

    with base_config.open() as file:
        resolved_config = yaml.safe_load(file)
    if chunk_size is not None:
        resolved_config["Model"]["chunk_size"] = chunk_size
        resolved_config["Model"]["overlap"] = overlap if overlap is not None else chunk_size // 2
    elif overlap is not None:
        resolved_config["Model"]["overlap"] = overlap
    if loop_enable is not None:
        resolved_config["Model"]["loop_enable"] = loop_enable
    if process_res is not None:
        resolved_config["Model"]["process_res"] = process_res

    # The upstream config uses paths relative to its own directory. A generated
    # config lives in workdir, so make the required weight locations absolute.
    for name, weight_path in resolved_config["Weights"].items():
        path = Path(weight_path)
        if not path.is_absolute():
            resolved_config["Weights"][name] = str((streaming_dir / path).resolve())

    workdir.mkdir(parents=True, exist_ok=True)
    derived_config = workdir / "streaming_config.yaml"
    with derived_config.open("w") as file:
        yaml.safe_dump(resolved_config, file, sort_keys=False)
    return derived_config


def _normalize_outputs(output_dir: Path, recon_dir: Path) -> Path:
    source_cloud = output_dir / "pcd" / "combined_pcd.ply"
    if not source_cloud.is_file():
        source_cloud = output_dir / "output.ply"
    if not source_cloud.is_file():
        raise FileNotFoundError(f"DA3-Streaming produced no point cloud in {output_dir}")

    destination = recon_dir / "fused_points.ply"
    shutil.copy2(source_cloud, destination)
    _assemble_depth_npz(output_dir, recon_dir)
    return destination


def _assemble_depth_npz(output_dir: Path, recon_dir: Path) -> None:
    """Create the optional depth archive used by the existing viz stage."""
    depth_files = sorted((output_dir / "results_output").glob("**/depth*.npy"))
    if not depth_files:
        return
    try:
        depths = np.stack([np.load(depth_file) for depth_file in depth_files])
        np.savez_compressed(recon_dir / "recon.npz", depth=depths)
    except (OSError, ValueError) as error:
        log.warning("Could not assemble streaming depth maps for visualization: %s", error)