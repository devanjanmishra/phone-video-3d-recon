# Licensing

Read this before commercial use — **the code license and the weights license are
different**, and one vendored component is GPL.

## Code

**This repository's code: Apache-2.0.** Free for commercial use, with an explicit
patent grant.

## Model weights

- **The default checkpoint (`DA3NESTED-GIANT-LARGE-1.1`): CC BY-NC 4.0 —
  non-commercial only.** It is the default because it is the only one giving true
  metric scale. Apache-2.0 code does **not** launder a non-commercial weights
  license.
- **For commercial use**, pass an Apache-2.0 checkpoint:
  `--model depth-anything/DA3-BASE`. You lose accuracy and absolute metric scale.
  `MapAnything` is the alternative worth evaluating: permissive weights *and*
  metric.
- No weights are vendored or redistributed here; they are fetched at runtime from
  their original hosts under their original terms. Verify the current license on
  the model card yourself — they can change independently of this project.

## Vendored source code

- `depth_anything_3/` — DA3 core, **Apache-2.0**, committed to this repo
  (unmodified upstream, license retained).
- `Depth-Anything-3/da3_streaming/` — DA3-Streaming, **Apache-2.0**, **not
  committed** (git-ignored; fetch it yourself per [STREAMING.md](STREAMING.md)).
- **SALAD**, the loop-closure descriptor bundled inside DA3-Streaming, is
  **GPL-3.0**.

## The SALAD / GPL-3.0 boundary

GPL-3.0 is strong copyleft. Vendoring SALAD into an Apache-2.0 repo and
distributing the combined work would create a licensing conflict. This is handled
by keeping the entire `Depth-Anything-3/` tree **out of the committed repo** (it
is git-ignored). Consequences:

- SALAD lives only in *your local working copy* after you fetch DA3-Streaming — it
  is not part of this project's distributed, Apache-2.0 tree.
- SALAD is only reached by the optional loop-closure path. Running streaming with
  `--no-loop` never imports it.
- If you ever choose to redistribute a tree that includes SALAD, you must treat
  that combined work as **GPL-3.0** and stop advertising it as Apache-2.0-only.

See [`../NOTICE`](../NOTICE) for the full third-party breakdown.
