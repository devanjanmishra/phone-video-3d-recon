# video2cad

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
```

**On "stitching":** there isn't an explicit one. DA3 processes all keyframes in a
single forward pass with global cross-view attention, so correspondence and
registration are learned, not post-hoc — every frame's pose comes out in one
shared world frame. Fusion is then just back-projection plus a 1 cm voxel merge.
No ICP, no pose graph, no loop closure. See [docs/PIPELINE.md](docs/PIPELINE.md).

## Install (Windows)

Requires Python 3.10+ and an NVIDIA GPU (8 GB VRAM minimum, 16–24 GB comfortable).

```powershell
python -m venv venv
venv\Scripts\activate

# 1. torch for YOUR CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install xformers

# 2. DA3 backbone
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3; pip install -e .; cd ..

# 3. this pipeline
pip install -e .
```

No compiler required — everything is prebuilt wheels. That is precisely why this
stack beats building COLMAP from source or fighting MASt3R-SLAM on Windows.

## Usage

```powershell
# full run
video2cad --video C:\videos\home.mp4 --workdir C:\recon\myhome

# low VRAM (8 GB): smaller model, fewer frames
video2cad --video home.mp4 --workdir out --model depth-anything/DA3-LARGE-1.1 --target-frames 120 --batch-limit 80

# re-tune CAD only, no GPU rerun
video2cad --workdir out --stages cad --slice-height 1.0
```

Outputs in `<workdir>`: `recon/fused_points.ply`, `cad/aligned_points.ply`,
`cad/planes_colored.ply`, `cad/planes.json`, `cad/mesh.obj|.stl`, `cad/house_plan.dxf`.

### Key flags

| Flag | Effect |
|---|---|
| `--target-frames` | more = denser and slower; 150–250 for a 2–3 BHK |
| `--conf-percentile` | raise (e.g. 55) to drop noisy geometry, at density cost |
| `--voxel` | 0.005 for fine detail, 0.02 for lighter CAD-oriented clouds |
| `--max-depth` | clip window/mirror hallucinations (default 12 m) |
| `--slice-height` | floor-plan cut height above floor (default 1.2 m) |
| `--model` | `DA3-BASE` for the Apache-2.0 path (see [Licensing](#licensing)) |

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
- **For commercial use**, pass an Apache-2.0 checkpoint: `--model depth-anything/DA3-BASE`. You lose accuracy and absolute metric scale. `MapAnything` is the alternative worth evaluating: permissive weights *and* metric.
- No weights are vendored or redistributed here; they are fetched at runtime from their original hosts under their original terms. Verify the current license on the model card yourself — they can change independently of this project.

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

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance Seed) ·
[Open3D](https://github.com/isl-org/Open3D) · [ezdxf](https://github.com/mozman/ezdxf) · [OpenCV](https://github.com/opencv/opencv)
