# Usage reference

Full flag reference, VRAM tables, and the CAD workflow. For a quickstart, see the
[README](../README.md). For the streaming setup, see [STREAMING.md](STREAMING.md).

## Getting a recommendation

```bash
python recommend.py --video your_video.mp4
```

Prints the optimal model, resolution, frame budget, and a ready-to-run command
based on your GPU and video.

## Run scripts

```bash
# Single-pass (all frames in one forward pass — best quality, limited by VRAM)
python run_single.py home.mp4 out_single --rescale-height 2.1
python run_single.py home.mp4 out_single --low-res --rescale-height 2.1

# Streaming (recommended for long videos; upstream DA3-Streaming)
python run_streaming.py home.mp4 out_stream --chunk-size 10 --process-res 336 --no-loop --rescale-height 2.1
```

## Direct CLI

```bash
# Single-pass run (short captures only)
video2cad --video home.mp4 --workdir output \
  --model depth-anything/DA3-SMALL \
  --target-frames 16 --max-frames 16 --process-res 336 \
  --rescale-height 2.1

# Long capture (DA3-Streaming)
video2cad --video home.mp4 --workdir output --stages frames,stream,cad,viz \
  --target-frames 300 --chunk-size 10 --process-res 336 --no-loop \
  --rescale-height 2.1

# Re-tune CAD only, no GPU rerun
video2cad --workdir output --stages cad --slice-height 1.0 --rescale-height 2.1

# Generate visualization GIFs only
video2cad --workdir output --stages viz
```

Outputs in `<workdir>`: `recon/fused_points.ply`, `cad/aligned_points.ply`,
`cad/planes_colored.ply`, `cad/planes.json`, `cad/mesh.obj|.stl`,
`cad/house_plan.dxf`, `viz/turntable.gif`, and, for non-streaming runs,
`viz/depth_montage.gif`.

## Key flags

| Flag | Effect |
|---|---|
| `--target-frames` | more = denser and slower; 60–120 for a room, 150–250 for a flat |
| `--max-frames` | maximum keyframes allowed for the single-pass path; use streaming above this limit |
| `--process-res` | DA3 internal resolution (504/378/336); lower = less VRAM, more frames |
| `--rescale-height` | known height in meters (door=2.1, ceiling=2.7) for non-metric models |
| `--conf-percentile` | raise (e.g. 55) to drop noisy geometry, at density cost |
| `--voxel` | 0.005 for fine detail, 0.02 for lighter CAD-oriented clouds |
| `--max-depth` | clip window/mirror hallucinations (default 12 m) |
| `--slice-height` | floor-plan cut height above floor (default 1.2 m) |
| `--model` | `DA3-SMALL` (default), `DA3-LARGE-1.1`, `DA3-BASE`, or `DA3NESTED-GIANT-LARGE-1.1` |

## Single-pass VRAM limits (tested on V100 16 GB, fp32)

| Model | `--process-res` | Max `--max-frames` | Quality |
|---|---|---|---|
| DA3-SMALL | 504 | 16 | Sharpest per-frame |
| DA3-SMALL | 378 | 32 | Good balance |
| **DA3-SMALL** | **336** | **40** | Highest single-pass coverage |
| DA3-LARGE | 504 | 8 | Best depth quality |
| DA3-LARGE | 336 | 20 | Large model + more frames |

> **Ampere+ GPUs (RTX 30xx/40xx/A100)** use bf16, roughly doubling these limits.
> A 24 GB RTX 4090 can do ~80 frames at 504px with DA3-LARGE in one pass.

## Reconstruction paths

| | Single pass | Streaming |
|---|---|---|
| **Frames** | Limited by VRAM | Unlimited |
| **Quality** | Best when every selected frame fits at once | Upstream chunk alignment + optional loop closure for long sequences |
| **When to use** | Short videos, high VRAM | Long / whole-home captures, or anything above the single-pass limit |
| **Setup** | Base dependencies | Extra weights and `.[streaming]` deps — see [STREAMING.md](STREAMING.md) |

## Tested configurations (20 s portrait home video, V100 16 GB)

| Config | Frames | Points | Walls | Floor dims | DXF |
|---|---|---|---|---|---|
| DA3-LARGE, 504px, single 8fr | 8 | 15,839 | 0 | 0.6×1.2 m | No |
| DA3-SMALL, 504px, single 16fr | 16 | 64,968 | 3 | 1.5×1.7 m | No |
| DA3-SMALL, 336px, single 40fr | 40 | 36,176 | 5 | 4.8×6.0 m | Yes |
| **DA3-Streaming, 336px, 10fr chunks** | **300** | **415,670** | **14** | — | **Yes** |

More frames increase coverage. Use DA3-Streaming for any capture that does not fit
in one forward pass.

## Rough-CAD workflow

- **`house_plan.dxf`** → FreeCAD (Draft workbench) or AutoCAD. Polylines are in meters on layer `WALLS`. Trace over them for a clean plan.
- **`planes.json`** → every wall/floor/ceiling with plane equation, centroid, bbox and `extent_m`. Fastest route to real dimensions.
- **`mesh.stl`** → FreeCAD: Mesh workbench → import → Part → *Shape from mesh* → solid for boolean/massing work. Or Blender to clean up.
- **Photorealistic twin (optional)** → point gsplat/nerfstudio at `recon/colmap/`: `ns-train splatfacto --data recon/colmap`.

## Accuracy

Values below are read from the committed [`../examples/planes.json`](../examples/planes.json)
(a real home-interior capture). The **ground-truth column still needs a tape
measure** — that is the one number that turns this from a demo into a validated
tool, and it's left blank deliberately rather than guessed.

| Feature | Ground truth | `planes.json` | Error |
|---|---|---|---|
| Longest wall run | _measure_ m | 6.01 m | _TBD_ |
| Room footprint (largest floor bbox) | _measure_ m | 6.24 × 5.52 m | _TBD_ |
| Floor-to-ceiling height (median of 14 walls) | _measure_ m | 2.14 m | _TBD_ |

**Internal consistency check** (needs no tape): the floor-to-ceiling height,
estimated independently from all 14 wall planes, lands at 2.09–2.30 m with a
median of 2.14 m. A typical Indian apartment ceiling is ~2.7 m and a low one
~2.4 m, so this run reads a bit low — consistent with the checkpoint used being
up-to-scale rather than the metric NESTED one. Pass `--rescale-height` with a
known height to correct it before trusting absolute dimensions.

Method: measure one wall with a tape, compare against `extent_m` of the
corresponding plane in `cad/planes.json`. List the capture and model checkpoint
with the numbers, since scale accuracy depends on both.

> Metric scale comes **only** from the nested checkpoint
> (`DA3NESTED-GIANT-LARGE-1.1`). With any other checkpoint the reconstruction is
> up-to-scale: geometry is correct in shape but the absolute meters are not, and
> you must rescale using one known measurement before the DXF dimensions mean
> anything.
