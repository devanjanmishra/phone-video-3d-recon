# home3d — Home video → metric 3D point cloud → rough CAD

Offline (non-real-time) monocular reconstruction pipeline. Input: a single walkthrough
video of an interior. Output: metric-scale point cloud, plane segmentation
(walls / floor / ceiling with real dimensions), watertight-ish mesh (OBJ/STL),
and a 2D floor-plan DXF that opens in AutoCAD / FreeCAD / LibreCAD.

## Why this stack (state of the art, mid-2026)

| Candidate | Verdict for home video, offline, Windows |
|---|---|
| **Depth Anything 3 (DA3)** | **Chosen.** Current SOTA feed-forward geometry: beats VGGT by ~44% pose / ~25% geometry accuracy. Nested model outputs **metric depth (meters)** → measurable CAD. Pure PyTorch → runs on Windows. DA3-Streaming handles very long videos in <12 GB VRAM. Code Apache-2.0 (top checkpoints CC BY-NC — fine for personal use; use DA3-BASE/SMALL for fully Apache). |
| VGGT (Meta) | Prior SOTA, now superseded by DA3; non-commercial license. |
| MapAnything (Meta) | Good, Apache-2.0 model available, metric — solid fallback; slightly behind DA3 on indoor benchmarks. |
| MASt3R-SLAM / VGGT-SLAM | Real-time focus (you don't need it); Linux-first builds, painful on Windows. |
| COLMAP (+3DGS) | Classical gold standard, official Windows binaries — but frequently **fails on textureless indoor walls**, and takes hours-to-days. Kept as optional refinement: this pipeline exports DA3 poses in COLMAP format so you can train Gaussian Splatting (gsplat / nerfstudio) on top for photorealism. |

## Pipeline

```
video.mp4
  └─ frames  : sharpness-ranked keyframe extraction (blur rejection)   [CPU]
  └─ recon   : DA3 → poses + intrinsics + metric depth + confidence    [GPU]
               → confidence-filtered fusion → fused_points.ply
               → COLMAP-format export (optional 3DGS later)
  └─ cad     : gravity alignment (floor → z=0)                          [CPU]
               → iterative RANSAC planes → planes.json (wall sizes in m)
               → Poisson mesh → mesh.obj / mesh.stl
               → horizontal slice → house_plan.dxf (WALLS layer)
```

## Windows setup

1. Install **Python 3.11+** (python.org, check "Add to PATH") and **NVIDIA driver + CUDA-capable GPU** (8 GB VRAM min, 16–24 GB comfortable).
2. In PowerShell:
```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install xformers
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3; pip install -e .; cd ..
pip install -r requirements.txt        # numpy opencv open3d ezdxf
```
No compiler needed — everything is prebuilt wheels (this is why DA3 beats
COLMAP-source or MASt3R-SLAM for Windows friction).

## Usage

```powershell
# full run
python pipeline.py --video C:\videos\home.mp4 --workdir C:\recon\myhome
python pipeline.py --video C:\usecase_development\gesture_detection\3d_recon\home_interiors.mp4 --workdir C:\usecase_development\gesture_detection\3d_recon

# low VRAM (8 GB): smaller model + fewer frames
python pipeline.py --video home.mp4 --workdir out --model depth-anything/DA3-LARGE-1.1 --target-frames 120 --batch-limit 80

# tweak CAD only (no GPU rerun)
python pipeline.py --workdir out --stages cad --slice-height 1.0
```

Outputs land in `<workdir>`: `recon/fused_points.ply`, `cad/aligned_points.ply`,
`cad/planes_colored.ply`, `cad/planes.json`, `cad/mesh.obj|.stl`, `cad/house_plan.dxf`.

## Rough-CAD workflow

- **house_plan.dxf** → open in FreeCAD (Draft workbench) or AutoCAD; polylines are in meters on layer `WALLS`. Trace over them for a clean plan.
- **planes.json** → each wall/floor/ceiling has plane equation, centroid, bbox and `extent_m` (e.g. wall 4.02 m × 2.6 m) — direct dimension source.
- **mesh.stl** → FreeCAD: Mesh workbench → import → Part → *Shape from mesh* → solid for boolean/rough massing. Or Blender for cleanup.
- Photorealistic twin (optional): point gsplat / nerfstudio at `recon/colmap/` (`ns-train splatfacto --data recon/colmap`).

## Capture tips (accuracy lives or dies here)

- Walk **slowly**, ~1 m/s, landscape, 4K if possible; overlap rooms via doorways in one continuous take.
- Turn on all lights; avoid pure white-wall close-ups (give the camera edges/corners).
- Two passes per room at different heights (~1.0 m and ~1.7 m) noticeably improves ceilings/floors.
- Sequences >≈160 keyframes: use DA3-Streaming (repo: `da3_streaming/`) then feed its output PLY straight into `--stages cad`.

## Knobs that matter for accuracy

| Flag | Effect |
|---|---|
| `--target-frames` | more frames = denser, slower; 150–250 for a 2-3BHK |
| `--conf-percentile` | raise (e.g. 55) to drop noisy geometry, at density cost |
| `--voxel` | 0.005 for fine detail, 0.02 for lighter CAD-oriented clouds |
| `--max-depth` | clip window/mirror hallucinations (default 12 m) |
| `use_ray_pose` (on by default in code) | slower, more accurate poses |
