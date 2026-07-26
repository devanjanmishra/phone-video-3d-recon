# Vendored source — Depth Anything 3

This directory is a **verbatim, unmodified copy** of the `src/depth_anything_3/`
package from:

    https://github.com/ByteDance-Seed/Depth-Anything-3

Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
Licensed under the Apache License, Version 2.0 — see `LICENSE` in this
directory for the full text.

## Why vendored

DA3 is not published on PyPI. Vendoring removes a fragile `git clone` +
`pip install -e .` step from setup, which matters most on Windows where users
are least likely to have a working build toolchain.

## Provenance

| | |
|---|---|
| Upstream path | `src/depth_anything_3/` |
| Vendored path | `depth_anything_3/` (repo root) |
| Modifications | **None.** All original copyright and license headers retained. |
| Upstream commit | _record the SHA you vendored from here_ |

The only change is the directory's position in the tree (`src/`-layout →
flat layout), so that `import depth_anything_3` works from a repo checkout.

## Updating

```bash
git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 /tmp/da3
rm -rf depth_anything_3
cp -r /tmp/da3/src/depth_anything_3 depth_anything_3
cp /tmp/da3/LICENSE depth_anything_3/LICENSE
git -C /tmp/da3 rev-parse HEAD    # record this SHA in the table above
```

If you ever **do** modify a vendored file, Apache-2.0 §4(b) requires you to
carry prominent notices stating that you changed it. Record each change here.

## Model weights

No model weights are vendored. They are downloaded at runtime from Hugging Face
under their own licenses, which differ per checkpoint and are **not** covered by
the Apache-2.0 license of this source. See the root `NOTICE`.
