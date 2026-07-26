# phone-video-3d-recon

**Turn a single handheld home walkthrough video into a metric 3D point cloud, segmented walls/floors, and a rough CAD floor plan (DXF). Offline, Windows-friendly, built on Depth Anything 3.**

[![License](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Weights](https://img.shields.io/badge/default%20weights-CC%20BY--NC-orange.svg)](#licensing)

<!-- TODO: replace with real renders before making the repo public. Three images,
     equal height, one row. This is the single highest-leverage thing in the repo. -->
| Fused point cloud | Segmented planes | Floor plan (DXF) |
|:---:|:---:|:---:|
| ![fused cloud](examples/01_fused_cloud.png) | ![planes](examples/02_planes_colored.png) | ![dxf](examples/03_floor_plan_dxf.png) |
| `recon/fused_points.ply` | `cad/planes_colored.ply` | `cad/house_plan.dxf` in FreeCAD |

No depth sensor, no LiDAR, no turntable, no COLMAP. One phone video in, measurable geometry out.

The Python package is `video2cad`; the repo is `phone-video-3d-recon`.

---

## Why this stack (state of the art, mid-2026)

| Candidate | Verdict for home video, offline, Windows |
|---|---|
| **Depth Anything 3 (DA3)** | **Chosen.** Current SOTA feed-forward geometry: ~44% better pose / ~25% better geometry accuracy than VGGT. The nested checkpoint outputs **metric depth in meters** → measurable CAD, not just up-to-scale shape. Pure PyTorch, prebuilt wheels → installs on Windows without a compiler. Code Apache-2.0. |
| VGGT (Meta) | Prior SOTA, superseded by DA3; non-commercial license. |
| MapAnything (Meta) | Strong, Apache-2.0 weights available, metric — the best fallback if you need permissive weights *and* metric scale. Slightly behind DA3 indoors. |
| MASt3R-SLAM / VGGT-SLAM | Optimized for real-time, which you don't need offline. Linux-first builds, painful on Windows. |
| COLMAP (+3DGS) | Classical gold standard with official Windows binaries — but routinely **fails on textureless interior walls** and takes hours to days. Kept as an *optional downstream*: this pipeline exports DA3 poses in COLMAP format so you can train Gaussian Splatting on top for photorealism. |

## Pipeline

```
video.mp4
  └─ frames  : sharpness-ranked keyframe extraction (blur rejection)   [CPU]
  └─ recon   : DA3 → poses + intrinsics + metric depth + confidence    [GPU]
               → confidence-filtered fusion → fused_points.ply
               → COLMAP-format export (optional 3DGS later)
  └─ cad     : gravity alignment (floor → z=0)                         [CPU]
               → iterative RANSAC planes → planes.json (wall sizes in m)
               → Poisson mesh → mesh.obj / mesh.stl
               → horizontal slice → house_plan.dxf (WALLS layer)
  └─ viz     : turntable point-cloud GIF + depth-montage GIF           [CPU]
```

**On "stitching":** there isn't an explicit one. DA3 processes all keyframes in a
single forward pass with global cross-view attention, so correspondence and
registration are learned, not post-hoc — every frame's pose comes out in one
shared world frame. Fusion is then just back-projection plus a 1 cm voxel merge.
No ICP, no pose graph, no loop closure. See [docs/PIPELINE.md](docs/PIPELINE.md).

## Install

Requires Python 3.10+ and an NVIDIA GPU (8 GB VRAM minimum, 16–24 GB comfortable).

```bash
git clone https://github.com/devanjanmishra/phone-video-3d-recon
cd phone-video-3d-recon
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. torch for YOUR CUDA version (check with nvidia-smi)
#    CUDA 12.1 (driver 535+):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install xformers --index-url https://download.pytorch.org/whl/cu121
#    CUDA 12.4 (driver 550+):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    pip install xformers

# 2. this pipeline (DA3 is vendored - no separate clone needed)
pip install -e .
```

> **Note:** Depth Anything 3 source is vendored inside this repo under
> `depth_anything_3/`. No separate `git clone` of DA3 is needed. Model weights
> are fetched at runtime from Hugging Face.

## Usage

### Quick start — get a recommendation

```bash
python recommend.py --video your_video.mp4
```

This prints the optimal model, resolution, batch size, and a ready-to-run command based on your GPU and video.

### Run scripts

```bash
# Single-pass (all frames in one forward pass — best quality, limited by VRAM)
python run_single.py home.mp4 out_single --rescale-height 2.1
python run_single.py home.mp4 out_single --low-res --rescale-height 2.1

# Batch mode (unlimited frames — processes overlapping chunks, aligns with Sim3)
python run_batch.py home.mp4 out_batch --rescale-height 2.1
python run_batch.py home.mp4 out_batch --high-model --rescale-height 2.1
python run_batch.py home.mp4 out_batch --target-frames 200 --rescale-height 2.1
```

### Direct CLI

```bash
# Full run (auto-batches if frames > batch-limit)
video2cad --video home.mp4 --workdir output \
  --model depth-anything/DA3-SMALL \
  --target-frames 120 --batch-limit 16 --process-res 336 \
  --rescale-height 2.1

# Re-tune CAD only, no GPU rerun
video2cad --workdir output --stages cad --slice-height 1.0 --rescale-height 2.1

# Generate visualization GIFs only
video2cad --workdir output --stages viz
```

Outputs in `<workdir>`: `recon/fused_points.ply`, `cad/aligned_points.ply`,
`cad/planes_colored.ply`, `cad/planes.json`, `cad/mesh.obj|.stl`, `cad/house_plan.dxf`,
`viz/turntable.gif`, `viz/depth_montage.gif`.

> The `viz` stage renders offscreen via Open3D. On a headless Linux box this
> needs an EGL-capable Open3D build; if it fails the stage logs the error and
> the rest of the pipeline still completes. It works out of the box on Windows.

### Key flags

| Flag | Effect |
|---|---|
| `--target-frames` | more = denser and slower; 60–120 for a room, 150–250 for a flat |
| `--batch-limit` | frames per forward pass; set below VRAM limit, auto-batches the rest |
| `--process-res` | DA3 internal resolution (504/378/336); lower = less VRAM, more frames |
| `--rescale-height` | known height in meters (door 2.1, ceiling 2.7). **Required for every checkpoint except the NESTED ones** - without it the DXF is not in meters. |
| `--conf-percentile` | raise (e.g. 55) to drop noisy geometry, at density cost |
| `--voxel` | 0.005 for fine detail, 0.02 for lighter CAD-oriented clouds |
| `--max-depth` | clip window/mirror hallucinations (default 12 m) |
| `--slice-height` | floor-plan cut height above floor (default 1.2 m) |
| `--model` | CLI default is `DA3NESTED-GIANT-LARGE-1.1` (metric, needs ~24 GB). The `run_single.py` / `run_batch.py` wrappers default to `DA3-SMALL` instead, because that is what fits 16 GB. Also `DA3-LARGE-1.1`, `DA3-BASE`. |

### VRAM limits (tested on V100 16GB, fp32)

| Model | `--process-res` | Max `--batch-limit` | Quality |
|---|---|---|---|
| DA3-SMALL | 504 | 16 | Sharpest per-frame |
| DA3-SMALL | 378 | 32 | Good balance |
| **DA3-SMALL** | **336** | **40** | **Best for batched mode** |
| DA3-LARGE | 504 | 8 | Best depth quality |
| DA3-LARGE | 336 | 20 | Large model + batch |

> **Ampere+ GPUs (RTX 30xx/40xx/A100)** use bf16, roughly doubling these limits.
> A 24 GB RTX 4090 can do ~80 frames at 504px with DA3-LARGE in one pass.

### Single-pass vs Batch mode

| | Single pass | Batch mode |
|---|---|---|
| **Frames** | Limited by VRAM | Unlimited |
| **Quality** | Best — global attention sees all frames | Good — Sim(3) alignment at chunk boundaries |
| **When to use** | Short videos, high VRAM | Long videos, limited VRAM |
| **Seams** | None | Possible at chunk boundaries |

### Tested configurations (20s portrait home video, V100 16GB)

| Config | Frames | Points | Walls | Floor dims | DXF |
|---|---|---|---|---|---|
| DA3-LARGE, 504px, single 8fr | 8 | 15,839 | 0 | 0.6×1.2m | No |
| DA3-SMALL, 504px, single 16fr | 16 | 64,968 | 3 | 1.5×1.7m | No |
| DA3-SMALL, 336px, single 40fr | 40 | 36,176 | 5 | 4.8×6.0m | Yes |
| **DA3-SMALL, 336px, batch 60fr** | **60** | **103,229** | **9** | — | **Yes** |

More frames = more coverage = better geometry. Batch mode removes the VRAM ceiling.

## Rough-CAD workflow

- **`house_plan.dxf`** → FreeCAD (Draft workbench) or AutoCAD. Polylines are in meters on layer `WALLS`. Trace over them for a clean plan.
- **`planes.json`** → every wall/floor/ceiling with plane equation, centroid, bbox and `extent_m`. This is your fastest route to real dimensions.
- **`mesh.stl`** → FreeCAD: Mesh workbench → import → Part → *Shape from mesh* → solid for boolean/massing work. Or Blender to clean up.
- **Photorealistic twin (optional)** → point gsplat/nerfstudio at `recon/colmap/`: `ns-train splatfacto --data recon/colmap`.

## Capture tips

Accuracy lives or dies here. See [docs/CAPTURE.md](docs/CAPTURE.md) for the long version.

- Walk **slowly** (~1 m/s), one continuous take, overlap rooms through doorways.
- All lights on. Avoid filling the frame with blank wall — give the camera corners and edges.
- Two passes per room at different heights (~1.0 m and ~1.7 m) noticeably improves floors and ceilings.
- **Landscape**, not portrait (see Limitations).
- Aim for 60–120 s for a room, 4–8 min for a flat.

## Accuracy

<!-- TODO: fill this in from your own tape measure before publishing. An honest
     measured number here is worth more than any benchmark citation, and almost
     nobody in this space publishes one. -->

Measured against a tape on a real capture:

| Feature | Ground truth | `planes.json` | Error |
|---|---|---|---|
| Living room wall, long | _TBD_ m | _TBD_ m | _TBD_ |
| Living room wall, short | _TBD_ m | _TBD_ m | _TBD_ |
| Floor-to-ceiling height | _TBD_ m | _TBD_ m | _TBD_ |

Method: measure with a tape, compare against `extent_m` of the corresponding
plane in `cad/planes.json`. Capture and model checkpoint used are listed with
the numbers, since scale accuracy depends on both.

> Metric scale comes **only** from the nested checkpoint
> (`DA3NESTED-GIANT-LARGE-1.1`). With any other checkpoint the reconstruction is
> up-to-scale: geometry is correct in shape but the absolute meters are not, and
> you must rescale using one known measurement (e.g. a door height) before the
> DXF dimensions mean anything.

## Limitations

Honest list. Read it before trusting a number.

- **Short captures leave holes.** Geometry only exists where the camera looked. A 20 s clip gives one partial room; unseen surfaces are simply absent, and the DXF will not close into a valid loop unless you actually orbited the space.
- **Mirrors, windows and glossy TVs produce garbage depth.** The model reconstructs the *reflected/transmitted* scene as if it were real geometry — phantom rooms behind walls. Mitigate with `--max-depth`, or mask them out.
- **The DXF assumes a single storey.** The floor plan is one horizontal slice at one height. Multi-floor captures collapse into a meaningless overlay; reconstruct each floor as a separate run.
- **Portrait video is off-distribution.** These models are trained overwhelmingly on landscape imagery. Portrait input runs, but pose and depth quality degrade. Shoot landscape.
- **Textureless walls remain the hard case.** Far better than COLMAP (which fails outright), but large blank surfaces are carried by the learned prior rather than by evidence — they can come out subtly bowed or misplaced.
- **Long captures hit an O(N²) attention ceiling.** ~160 keyframes per forward pass on 24 GB. Beyond that, DA3-Streaming processes overlapping windows and aligns them with a similarity transform — any seams or scale jumps in a whole-house capture will appear at those window boundaries.
- **No semantic understanding.** Furniture, clutter and people are fused as geometry like everything else. Empty the room if you want clean walls.
- **Not validated for anything that matters.** This is a rough-CAD helper, not a survey instrument. Do not use it for structural, legal, or contractual measurements.

## Licensing

Read this before commercial use — **the code license and the weights license are different**.

- **This repository's code: Apache-2.0.** Free for commercial use, with an explicit patent grant.
- **The default checkpoint (`DA3NESTED-GIANT-LARGE-1.1`): CC BY-NC 4.0 — non-commercial only.** It is the default because it is the only one giving true metric scale. Apache-2.0 code does **not** launder a non-commercial weights license.
- **`DA3-SMALL` / `DA3-BASE` are Apache-2.0** and are what the wrapper scripts use by default, so the common path here is already commercial-safe. Confirm on the model card before relying on it.
- **For commercial use**, pass an Apache-2.0 checkpoint: `--model depth-anything/DA3-BASE`. You lose accuracy and absolute metric scale. `MapAnything` is the alternative worth evaluating: permissive weights *and* metric.
- **DA3 source code is vendored** in `depth_anything_3/` (Apache-2.0, unmodified, license included there). No model *weights* are vendored; they are fetched at runtime from their original hosts under their original terms. Verify the current license on the model card yourself — they can change independently of this project.

See [NOTICE](NOTICE) for the full third-party breakdown.

## Citation

If this pipeline is useful in your work, cite the model it stands on:

```bibtex
@article{depthanything3,
  title  = {Depth Anything 3: Recovering the Visual Space from Any Views},
  author = {ByteDance Seed},
  journal= {arXiv preprint arXiv:2511.10647},
  year   = {2025}
}
```

## Acknowledgements

This project vendors [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(ByteDance Seed, Apache-2.0) as its reconstruction backbone. The vendored source
is in `depth_anything_3/` and is subject to the
[DA3 license](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE).
Model weights are downloaded at runtime from Hugging Face under their own terms
(see [Licensing](#licensing)).

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance Seed) ·
[Open3D](https://github.com/isl-org/Open3D) · [ezdxf](https://github.com/mozman/ezdxf) · [OpenCV](https://github.com/opencv/opencv)
