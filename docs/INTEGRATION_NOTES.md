# Integration notes — streaming-only upload

Reconciling the working `video2cad-streaming-only` upload against the GitHub
repo (`phone-video-3d-recon`). The upload is the authoritative working code;
this round is integration only — no logic was changed. Everything below was
verified by running it.

## What the upload is

A complete, working tree where the `stream` stage is functional. It supersedes
the previous scaffold: it vendors the **full DA3-Streaming implementation** (with
SALAD, the fastloop C++/Python Sim3 solver, and configs) under
`Depth-Anything-3/`, and ships a cleaner rewrite of `streaming_stage.py` that I
took as ground truth.

Verified end-to-end against a faithful mock of the DA3-Streaming binary (no GPU
or weights here): keyframe staging → derived config → subprocess launch →
`combined_pcd.ply` normalized to `recon/fused_points.ply` → CAD recovers 4 walls
/ 1 floor / 1 ceiling with correct extents and a valid DXF. The upload's
`streaming_stage.py` also makes weight paths absolute in the derived config,
which the previous version did not — a real improvement, kept as-is.

## What changed between GitHub and the upload

| File | Status | Action |
|---|---|---|
| `video2cad/streaming_stage.py`, `run_streaming.py` | new, working | kept |
| `Depth-Anything-3/` (DA3-Streaming + SALAD) | new, vendored | kept **but git-ignored** (see licensing) |
| `depth_anything_3/` core | present, Apache | kept, committed |
| `da3_stage.py`, `cli.py`, `cad_stage.py`, `visualize.py`, wrappers, README | edited | kept as uploaded |
| `run_batch.py` | **removed** in upload | left removed (you said batch goes later; the upload already did it) |
| `docs/INTEGRATION_NOTES.md` (old) | dropped in upload | replaced by this file |
| `home_interiors.mp4` | dropped in upload | left out — a 22 MB video doesn't belong in the source tree |

## Integration fixes applied (no logic touched)

1. **`pyproject.toml` URLs** pointed at `/video2cad` again (a regression). Fixed
   to `/phone-video-3d-recon` (Homepage / Repository / Issues).

2. **NOTICE was stale and wrong on a license-critical point.** It still claimed
   the project vendors nothing, while the upload vendors three source trees.
   Rewrote it to map each vendored tree to its license, and added a prominent
   **GPL-3.0 warning for SALAD** (see below).

3. **License files for vendored trees.** `depth_anything_3/` already had its
   Apache `LICENSE`; added the Apache `LICENSE` to the `Depth-Anything-3/` tree
   too. SALAD's GPL `LICENSE` was already present and is retained.

4. **Stray VCS metadata removed.** `Depth-Anything-3/.../salad/.git` was a
   dangling gitlink (`gitdir: ../../../.git/modules/...`) that would break a
   fresh commit. Removed, along with nested `.gitignore` files in vendored trees.

5. **README streaming setup corrected.** It said "this workspace already includes
   [DA3-Streaming] at `Depth-Anything-3/da3_streaming`" — true for your local
   copy, but **false for a fresh clone**, because that path is git-ignored.
   Rewrote the setup to show the one-time `git clone --recursive` +
   `DA3_STREAMING_DIR` step, and added a source-code licensing block covering the
   SALAD GPL boundary.

## The SALAD / GPL-3.0 situation — the one thing that mattered most

SALAD (the DINOv2-based place-recognition descriptor that DA3-Streaming uses for
loop closure) is **GPL-3.0**, a strong copyleft license. Vendoring it into an
Apache-2.0 repo and distributing the combined work would create a licensing
conflict.

**This is already handled correctly by the upload's `.gitignore`**, which
excludes the entire `Depth-Anything-3/` tree. Confirmed by simulation: a commit
of this tree includes the Apache `depth_anything_3/` core (95 files) and **zero**
SALAD or `Depth-Anything-3/` files. So SALAD lives only in your local working
copy for running the pipeline; it is not part of your distributed, Apache-2.0
repo. A fresh clone fetches DA3-Streaming separately (`git clone --recursive`),
so the GPL code enters the *user's* tree, not your distribution.

SALAD is only reached by the optional loop-closure path. `--no-loop` never
imports it. If you ever decide to commit `Depth-Anything-3/`, you must treat the
whole distributed work as GPL-3.0 and stop advertising the repo as Apache-2.0.

## What lands on GitHub

First-party code + docs + the Apache `depth_anything_3/` core (95 files).
**Excluded** by `.gitignore`: `Depth-Anything-3/` (53 MB, includes GPL SALAD),
`uv.lock`, `*.mp4`, reconstruction outputs. The provided zip already omits the
git-ignored `Depth-Anything-3/` tree, so it mirrors what a clean commit contains.

## Decisions left to you

1. **`run_batch.py` is gone.** You said batch removal was for later, but the
   upload already removed it and the CLI/wrappers no longer reference it. Left as
   removed. If you wanted to keep batch one more cycle, restore it from git
   history — but nothing depends on it now.

2. **`uv.lock` stays uncommitted** (git-ignored in the upload). For an
   application, committing the lock is the more common choice for reproducibility.
   Your call; I did not change the `.gitignore` rule.

3. **Accuracy table and example images** are still the two open placeholders,
   unchanged from before. A tape-measured wall vs `planes.json` and three result
   renders remain the highest-value additions before making the repo public.
