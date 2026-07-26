# Integration notes

Review of the uploaded changes before merging into
`github.com/devanjanmishra/phone-video-3d-recon`. Everything below was verified
by running it, not by reading it.

## Bugs found and fixed

### 1. Batch mode crashed on a single bad depth pixel — **blocker**

`_compute_sim3_alignment` back-projected each chunk's overlap frames through
`_backproject_frames`, which filtered by that chunk's own `(d > 0) & isfinite(d)`
mask. Umeyama needs *index-aligned* correspondences, so filtering each side
independently means:

- lengths differ → `ValueError` in the matmul (cryptic, mid-run, after minutes
  of GPU work), or
- lengths coincidentally match → correspondence silently shifts and the fitted
  `(s, R, t)` is garbage, welding the next chunk on at the wrong scale/pose.

Reproduced:

```
clean case                       → scale 0.5000 (truth 0.5)   OK
ONE zero-depth pixel in chunk B  → ValueError: size 2303 != 2304
whole frame invalid in chunk A   → ValueError: size 2304 != 1536
```

Fixed by returning points *plus* a validity mask without filtering, then
intersecting the masks before sampling. All three cases now recover 0.500000
exactly. Added a floor of 100 jointly-valid pixels with a clear error, and a
warning when the fitted scale lands outside 0.1–10.

The Umeyama math itself was correct — verified independently.

### 2. `--rescale-height` corrupted dimensions by ~12%

After rescaling the cloud to true meters, the code did:

```python
plane_dist_thresh *= scale
min_plane_points = max(500, int(min_plane_points / scale))
```

Both are wrong. Once rescaled, the cloud **is** in meters, so the 2 cm RANSAC
threshold applies as-is; multiplying it by `scale` re-breaks what the rescale
just fixed. And `min_plane_points` is a *count* — scaling coordinates neither
creates nor destroys points, so dividing it by `scale` is dimensionally
meaningless (with `scale=10` it silently dropped the threshold 4000 → 400).

Separately, `_align_to_gravity` ran *before* the rescale using a metric
threshold on a cloud that was still in arbitrary units.

Measured on a synthetic 4×5×2.6 m room fed in at 10× too small, with
`--rescale-height 2.6`:

| | 4 m walls recovered |
|---|---|
| before | 3.48, 3.56 (≈12% error) |
| after | 3.94, 3.95 |
| truth | 4.00, 4.00 |

Fixed: gravity alignment now uses a threshold relative to the cloud's bbox
diagonal (scale-invariant), and the metric thresholds are left alone after
rescaling. Output is now identical whether the input cloud arrives at 0.02×,
0.1×, 1× or 7× — verified across all four.

The residual ~1.5% under-measure is inherent (corner points get claimed by the
perpendicular wall's RANSAC), not a bug.

### 3. `run_single.py` was not single-pass

Defaults were `--target-frames 60` with `batch_limit` forced to 16 (or 40 with
`--low-res`). Since `video2cad` batches whenever `frames > batch_limit`, the
script that exists specifically to do a single forward pass was silently
chunking — so "single vs batch" comparisons were really batch vs batch.

Fixed: `batch_limit` now follows `target_frames`, and `--target-frames` defaults
to what actually fits (16 at 504px, 40 at 336px), matching the README's tested
table. Asking for more prints an explicit OOM warning and points at
`run_batch.py`. Added `--process-res` for parity with `run_batch.py`.

### 4. `compare.py` quality score exceeded its own maximum

`min(n_walls, 10) * 10` alone reaches 100, before the other five terms, all
printed as `/ 100`. Reweighted to sum to exactly 100
(walls 30, coverage 25, DXF 15, density 15, height consistency 10, floor 5).
Also `wall_height_std < 0.2` awarded 10 points when *no* walls were found
(std = 0); now requires a nonzero std.

### 5. `recommend.py` recommended commands it then overrode

It computed a `batch_limit` for the detected GPU, then emitted a `run_single.py`
/ `run_batch.py` command *without* passing it — and both wrappers reset it from
their own presets. A 48 GB GPU got "batch_limit 80" printed and 16 executed.
Now both numbers are always passed explicitly. Also removed the `CONFIGS` dict
(defined, never referenced) and fixed `if args.vram:` rejecting `--vram 0`.

### 6. `recon.npz` stores absolute frame paths

The depth-montage GIF silently skipped every frame if the workdir was moved or
produced on another machine. `depth_montage_gif` already took a `frames_dir`
argument but never used it; it is now the fallback lookup by filename.

## Legal / packaging

### Vendoring DA3 — compliance gap, now closed

Vendoring is the right call (removes a `git clone` + editable install from
setup, which is where Windows users bail). But Apache-2.0 §4 requires shipping
the license with the redistributed source, and `depth_anything_3/` had no
`LICENSE`.

Added `depth_anything_3/LICENSE` (upstream text) and
`depth_anything_3/VENDORED.md` recording provenance and update procedure. Root
`NOTICE` updated — it still claimed the project vendors nothing.

Verified against upstream: the vendored tree is **byte-identical** to
`src/depth_anything_3/` at `main` (checked `api.py`, `cli.py`, `registry.py`,
`specs.py`, `cfg.py`), and all 83 files retain their ByteDance copyright
headers. Because nothing was modified, no §4(b) "changed files" notice is
needed — but if you ever patch a vendored file, record it in `VENDORED.md`.

Fill in the upstream commit SHA in `VENDORED.md`; I could not determine which
commit you vendored from.

### README/code contradictions

- Install said `git clone .../video2cad`; the repo is `phone-video-3d-recon`.
  Fixed throughout, including `pyproject.toml` URLs.
- Flags table said `--model DA3-SMALL (default)`; the CLI default is
  `DA3NESTED-GIANT-LARGE-1.1`. Only the wrapper scripts default to `DA3-SMALL`.
  Documented exactly, without changing behaviour — see "Decisions for you".
- Licensing section said no vendoring; now distinguishes vendored *source*
  (Apache-2.0) from *weights* (per-checkpoint), and notes that the wrappers'
  `DA3-SMALL` default is already the commercial-safe path.
- `examples/` was dropped from the upload but exists in the repo — restored.

## Decisions for you

1. **Default checkpoint.** Code defaults to the metric-but-24 GB
   `DA3NESTED-GIANT-LARGE-1.1`; every config you have actually tested uses
   `DA3-SMALL`. I left the code alone and made the docs precise. If you'd rather
   the default be something your own GPU can run, change `DEFAULT_MODEL` in
   `video2cad/da3_stage.py` *and* the `--model` default in `video2cad/cli.py` —
   and then flip the CC BY-NC badge, since `DA3-SMALL` is Apache-2.0.

2. **`uv.lock` is gitignored but present in your tree.** For an application
   (which this is) committing the lock is the usual choice — it pins the exact
   dependency set that produced your tested numbers. 623 KB. Your call; I left
   your `.gitignore` as written.

3. **`home_interiors.mp4` (22 MB) is committed on `main`.** Under GitHub's
   100 MB limit, so it works, but every clone pays 22 MB forever and it's
   already in history. Options: leave it, move to Git LFS, or drop it and link a
   release asset. Removing it from history needs a rewrite — decide before the
   repo gets stars or forks.

4. **Package name stays `video2cad`** inside repo `phone-video-3d-recon`.
   Python packages can't contain hyphens, so they can't match exactly. Renaming
   the package to `phone_video_3d_recon` is possible but churns every import for
   no functional gain.

## Still open (not blocking)

- The accuracy table is still `_TBD_`. This is the highest-value empty slot in
  the repo — one tape-measured wall vs `planes.json` beats any benchmark
  citation. Note that fix #2 above changes these numbers, so measure *after*
  merging.
- `examples/01_fused_cloud.png`, `02_planes_colored.png`,
  `03_floor_plan_dxf.png` are referenced by the README and still missing, so
  the top of the page renders three broken images. You now have `viz/*.gif` —
  a turntable GIF would be a stronger hero image than a still.
- The `viz` stage needs an EGL-capable Open3D on headless Linux. It fails
  gracefully (logged, pipeline continues) and works on Windows; noted in the
  README.
