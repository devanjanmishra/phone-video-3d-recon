# Example outputs

Real artifacts from one home-interior capture, small enough to live in git.
The heavy outputs from the same run (`mesh.obj` 52 MB, `mesh.stl` 29 MB,
`aligned_points.ply` 11 MB, `planes_colored.ply` 7 MB) are **not** committed —
they're reproducible from the pipeline and would bloat every clone. See the
root `.gitignore`.

| File | What it is |
|---|---|
| `turntable.gif` | The reconstructed DA3 point cloud, orbiting (downscaled to 480px / 0.4 MB for the README). |
| `floor_plan.png` | The extracted `house_plan.dxf` rendered to an image. |
| `house_plan.dxf` | The actual 2D floor plan (24 KB). Opens in AutoCAD / FreeCAD / LibreCAD; polylines in meters on layer `WALLS`. |
| `planes.json` | The full plane segmentation (14 walls, floors, ceilings) with per-plane `extent_m` — the basis for the accuracy table in the root README. |

To regenerate for your own capture:

```bash
python run_streaming.py your_home.mp4 out --rescale-height 2.1   # long/loopy
# or
python run_single.py   your_home.mp4 out                        # short
```

The DXF, `planes.json`, mesh and clouds all land in `out/cad/`.
