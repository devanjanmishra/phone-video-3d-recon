# Design & background

Why this project is built the way it is: what it stands on, why Depth Anything 3
was chosen over the alternatives, and what it adds on top.

## Built on Depth Anything 3 — with thanks

This project is not a new reconstruction model. It is a practical pipeline that
stands entirely on the shoulders of **[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)**
(ByteDance Seed). DA3 does the hard part — recovering camera poses and dense
metric geometry from unposed images in a single feed-forward pass — and it does
it better than anything else available as of mid-2026. Everything here exists
because DA3 made that step reliable and fast enough to build on.

Concretely, this project uses DA3 for:

- **Pose + intrinsics estimation** across all keyframes in one shared world frame
  (no COLMAP, no SLAM front-end).
- **Dense per-pixel depth**, and — via the nested checkpoint — depth in true
  **meters**, which is what makes the downstream CAD dimensions meaningful.
- **The streaming path**, which wraps ByteDance's own DA3-Streaming (itself built
  on VGGT-Long) for long, memory-bounded captures with optional loop closure.

Full credit and licensing for DA3 and the other components is in
[`../NOTICE`](../NOTICE) and [`LICENSING.md`](LICENSING.md). If DA3 is useful in
your work, cite it (BibTeX in the README).

## Why DA3 over the alternatives (state of the art, mid-2026)

| Candidate | Verdict for home video, offline, Windows |
|---|---|
| **Depth Anything 3 (DA3)** | **Chosen.** Current SOTA feed-forward geometry: ~44% better pose / ~25% better geometry accuracy than VGGT. The nested checkpoint outputs **metric depth in meters** → measurable CAD, not just up-to-scale shape. Pure PyTorch, prebuilt wheels → installs on Windows without a compiler. Code Apache-2.0. |
| VGGT (Meta) | Prior SOTA, superseded by DA3; non-commercial license. |
| MapAnything (Meta) | Strong, Apache-2.0 weights available, metric — the best fallback if you need permissive weights *and* metric scale. Slightly behind DA3 indoors. |
| MASt3R-SLAM / VGGT-SLAM | Optimized for real-time, which you don't need offline. Linux-first builds, painful on Windows. |
| COLMAP (+3DGS) | Classical gold standard with official Windows binaries — but routinely **fails on textureless interior walls** and takes hours to days. Kept as an *optional downstream*: this pipeline exports DA3 poses in COLMAP format so you can train Gaussian Splatting on top for photorealism. |

## What this project adds on top of DA3

DA3 gives you poses and depth. It does **not** give you a CAD-ready deliverable,
a Windows-friendly one-command workflow, or the glue that makes a raw phone video
usable. That gap is what this project fills:

1. **Video → measurable CAD, end to end.** DA3 stops at geometry. This pipeline
   carries it all the way to a metric point cloud, a segmented set of
   walls/floors/ceilings with real dimensions (`planes.json`), a mesh
   (`mesh.obj`/`.stl`), and a 2D floor-plan **DXF** that opens in AutoCAD /
   FreeCAD / LibreCAD. Nobody else in this space closes the loop to DXF.

2. **Blur-aware keyframe selection.** Reconstruction quality is bottlenecked by
   motion blur, not frame count. The `frames` stage scores every candidate frame
   by variance-of-Laplacian and keeps the sharpest one per time window, instead
   of a blind stride. Feeding DA3 clean frames measurably improves the result.

3. **Metric-scale recovery for any checkpoint.** True meters come only from the
   nested checkpoint. For every other (faster, permissively-licensed) checkpoint,
   `--rescale-height` recovers absolute scale from a single known measurement
   (a door, a ceiling), so the DXF is still dimensionally usable.

4. **CAD-oriented geometry cleanup.** Gravity alignment (floor → z=0), iterative
   RANSAC plane extraction with semantic labeling, density-trimmed Poisson
   meshing, and a horizontal-slice → contour-trace → polyline-simplify path that
   turns a noisy cloud into straightened DXF walls.

5. **Three reconstruction paths behind one interface.** Single-pass (best
   quality, VRAM-bound), streaming (long/whole-home, loop closure), and a re-runnable
   CAD stage — all through the same CLI, so you can re-tune CAD parameters without
   paying for another GPU pass. See [USAGE.md](USAGE.md).

6. **Windows-first, compiler-free.** DA3 is vendored so there's no separate clone;
   everything installs from prebuilt wheels. This is a deliberate contrast to
   COLMAP-from-source and MASt3R-SLAM, which fight you on Windows.

7. **Honesty as a feature.** A published [limitations list](../README.md#limitations),
   an [accuracy method](USAGE.md#accuracy) with real numbers from a real capture,
   and an internal-consistency check that needs no ground truth. Almost no repo in
   this space publishes a measured error; this one is built to.

## The pipeline at a glance

```
video.mp4
  └─ frames  : sharpness-ranked keyframe extraction (blur rejection)   [CPU]
  └─ recon   : DA3 → poses + intrinsics + metric depth + confidence    [GPU]
               → confidence-filtered fusion → fused_points.ply
               → COLMAP-format export (optional 3DGS later)
     (or)
     stream  : DA3-Streaming → chunks + Sim(3) + optional loop closure  [GPU]
  └─ cad     : gravity alignment (floor → z=0)                         [CPU]
               → iterative RANSAC planes → planes.json (wall sizes in m)
               → Poisson mesh → mesh.obj / mesh.stl
               → horizontal slice → house_plan.dxf (WALLS layer)
  └─ viz     : turntable point-cloud GIF + depth-montage GIF           [CPU]
```

**On "stitching":** there is no explicit one. DA3 processes all keyframes in a
single forward pass with global cross-view attention, so correspondence and
registration are learned, not post-hoc — every frame's pose comes out in one
shared world frame. Fusion is then just back-projection plus a 1 cm voxel merge.
No ICP, no pose graph, no loop closure (except in the streaming path). The full
mechanism is in [PIPELINE.md](PIPELINE.md).
