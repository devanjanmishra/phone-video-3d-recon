# Pipeline internals

Three stages, each independently re-runnable via `--stages`. A `manifest.json`
in the workdir carries state between them, so you can re-tune CAD parameters
without paying for another GPU pass.

```
video.mp4 ──frames──▶ keyframes ──recon──▶ poses + metric depth ──cad──▶ PLY / planes.json / mesh / DXF
            [CPU]                 [GPU]                            [CPU]
```

---

## Stage 1 — `frames.py`

Reconstruction quality is bottlenecked by **motion blur**, not by frame count.
Blurred frames don't merely contribute nothing — they actively poison pose
estimation.

So instead of a blind stride (`every Nth frame`), the timeline is split into
`target_frames` equal windows, every frame in each window is scored by
**variance of the Laplacian** (sharp edges → high second-derivative variance),
and the sharpest frame per window is kept. You get temporally uniform coverage
*and* the best available frame at each position. Frames below `min_sharpness`
are dropped entirely; if more than ~30% are dropped, the capture was too fast
and should be re-shot.

Scoring runs on a quarter-scale grayscale copy — the metric is scale-tolerant
and this keeps the pass fast. Kept frames are resized so the long edge is
`--long-edge` (default 1008 px, DA3's comfortable range) and written as
quality-95 JPEG.

## Stage 2 — `da3_stage.py`

One DA3 forward pass over all keyframes returns four arrays:

| Output | Shape | Meaning |
|---|---|---|
| `depth` | `[N,H,W]` | per-pixel depth, **in meters** (nested checkpoint only) |
| `conf` | `[N,H,W]` | per-pixel confidence |
| `extrinsics` | `[N,3,4]` | world→camera `[R\|t]`, OpenCV/COLMAP convention |
| `intrinsics` | `[N,3,3]` | per-frame `K` |

### How frames get stitched together

**There is no explicit stitching step.** No feature matching, no ICP, no pose
graph, no loop closure. This is the biggest architectural difference from
classical pipelines, and it's worth understanding.

**1. Joint inference puts every frame in one shared coordinate system.**
DA3 does not process frames independently. All N frames are tokenized and passed
through a single transformer whose attention alternates between within-frame
layers and **global cross-view layers**, where every patch of every frame
attends to every patch of every other frame. That cross-attention is where
correspondence happens — implicitly, densely, as a learned soft operation rather
than as discrete keypoint matches. The network learns "this sofa corner in frame
12 is the same 3D point as that one in frame 47."

The result: poses emerge already expressed in one common world frame, and the
depth maps are *predicted to be mutually consistent* — frame 12's depth and
frame 47's depth, back-projected through their respective poses, land on the
same surface. **Registration is a training objective, not a post-processing
step.**

**2. Back-projection + voxel merge cashes it out.**
`fuse_point_cloud` does the standard pinhole inversion — for pixel (u,v) at
depth d:

```
x = (u - cx) · d / fx
y = (v - cy) · d / fy
z = d
```

giving a point in *camera* coordinates. The extrinsic is world→camera
(`x_cam = R·x_world + t`), so we invert it:

```
x_world = Rᵀ · (x_cam - t)          # the `(pts_cam - t) @ R` line
```

Because every frame's `R,t` live in the same world frame, all N clouds land
directly on top of each other. Overlap regions — a wall seen by 15 frames yields
15 near-coincident point sets — are merged by `voxel_down_sample(0.01)`: points
sharing a 1 cm voxel collapse to their centroid. **That voxel averaging is the
only fusion operation.** It's what turns 15 noisy copies of a wall into one
wall, and it's why residual misalignment shows up as wall *thickening* rather
than ghosting.

Confidence filtering runs first (drop the bottom `--conf-percentile`), removing
exactly the pixels where cross-view consistency was weakest — sky through
windows, mirrors, blurred edges — which would otherwise smear the merge.

### What this replaced

| Classical approach | Failure mode this avoids |
|---|---|
| **COLMAP**: keypoints → match → triangulate → bundle adjustment | Hours to days; fails outright on blank interior walls |
| **SLAM** (MASt3R-SLAM etc.): frame-to-frame tracking → drift → loop closure → pose graph | Drift accumulation; needs revisits to close loops |
| **Per-frame depth + ICP**: predict depth independently, align clouds iteratively | Monocular depth has **inconsistent scale per frame**, so ICP has nothing rigid to align |

Joint prediction dissolves all three at once: scale is shared (one network
state), drift is bounded (every frame attends to every other — effectively *all
pairs are loop closures*), and textureless walls are carried by the learned
prior rather than by evidence.

### Where explicit stitching does come back

Global attention is **O(N²) in frames**, so ~160 keyframes is the practical
24 GB VRAM ceiling. Beyond that, DA3-Streaming processes overlapping sliding
windows and *does* explicitly align consecutive windows — estimating a
similarity transform (rotation, translation, **scale**) from the frames the
windows share, chaining them into one map.

So: **within a window, stitching is implicit in attention; across windows, it's
classical Sim(3) alignment on the overlap.** Any seam or scale jump in a
whole-house capture appears at exactly those boundaries.

### Metric scale

The any-view model's world frame is internally consistent but only
**up-to-scale**. The *nested* checkpoint runs a separate monocular **metric**
depth head on the reference view and rescales the entire joint reconstruction to
it — one global scalar, applied once. That single scalar is why the DXF ends up
in true meters, and why no other checkpoint gives real dimensions.

### COLMAP export

Poses are also written as a COLMAP text model (`PINHOLE`, one camera per image)
so downstream Gaussian Splatting tooling can consume them:
`ns-train splatfacto --data recon/colmap`. No COLMAP code is used — it's an
interchange format only.

## Stage 3 — `cad_stage.py`

**Gravity alignment.** RANSAC finds the dominant horizontal plane and rotates
the scene so its normal is +Z. If most points sit *below* that plane, it was the
ceiling — the scene is flipped 180° about X. The floor is then anchored to z=0
using the 1st percentile of z (robust to stray points below floor level).

**Plane segmentation.** Iterative RANSAC: fit plane → record inliers → remove
them → repeat, until fewer than `min_plane_points` remain. Each plane is labeled
by normal direction and height:

| Condition | Label |
|---|---|
| `abs(n_z) > 0.9`, centroid below scene mid-height | `floor` |
| `abs(n_z) > 0.9`, centroid above mid-height | `ceiling` |
| `abs(n_z) < 0.2` | `wall` |
| otherwise | `slanted` |

Each entry in `planes.json` carries the plane equation, centroid, bbox and
`extent_m` — **this is the dimension source for CAD work**.

**Mesh.** Poisson reconstruction (depth=10), density-trimmed at the 3rd
percentile to cut the inflated halo, decimated to 600k triangles.

**Floor plan.** A 10 cm-thick horizontal slab at `--slice-height` above the
floor is rasterized to a 2 cm occupancy grid, morphologically closed, contour-
traced (OpenCV), polygon-simplified with `approxPolyDP` (this is what
straightens walls), and written as DXF polylines in **meters** on layer `WALLS`.

Slice height matters: 1.2 m is chosen to cut above furniture but below wall
cabinets. Lower it if rooms are cluttered at that height.
