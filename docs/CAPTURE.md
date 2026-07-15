# Capture guide

The single largest determinant of output quality is the video, not the flags.
Ten minutes of care here beats any amount of parameter tuning afterward.

## The rules that matter most

**1. Walk slowly — about 1 m/s.** Roughly half your natural walking pace. Motion
blur is the primary killer; the frame extractor rejects blurred frames, so a
fast walk simply means fewer usable frames, not worse ones you can salvage.

**2. Shoot landscape.** These models are trained overwhelmingly on landscape
imagery. Portrait runs but is off-distribution, and pose/depth quality degrades.

**3. One continuous take.** Move between rooms *through* doorways without
cutting. The doorway frames are what tie two rooms into one coordinate system —
if you stop and restart, you get two disconnected reconstructions.

**4. All lights on.** Plus daylight if available. Underexposed frames are noisy
frames.

**5. Never fill the frame with blank wall.** Keep corners, edges, furniture,
door frames in view — the geometry there is evidence-backed, whereas a
full-frame beige wall is the model guessing. Sweep *along* walls at an angle
rather than pointing straight at them.

**6. Two heights per room.** One pass at ~1.0 m and one at ~1.7 m. This is the
cheapest single improvement to floor and ceiling quality, because it gives real
vertical parallax.

**7. Rotate, don't just pan.** Pure rotation from a fixed spot gives the network
no translation baseline. Arc *around* the room; walk sideways past objects.

## Duration targets

| Subject | Target length | Suggested `--target-frames` |
|---|---|---|
| Single room | 60–90 s | 120–150 |
| 2 BHK flat | 4–6 min | 200–250, or DA3-Streaming |
| 3 BHK / house | 6–10 min | DA3-Streaming (exceeds one-pass limit) |

A 20 s clip yields one partial room with holes. It will reconstruct, but the DXF
won't close.

## Camera settings

- **4K if available**, 30 fps. Frames are downscaled to 1008 px long edge anyway,
  but a higher-resolution sensor read means less noise per downscaled pixel.
- **Lock exposure and white balance** if your phone allows it. Auto-exposure
  hunting between a bright window and a dark corridor creates frames the model
  has to reconcile.
- **Disable HDR** and any "cinematic"/beauty modes — they apply per-frame
  temporal processing that breaks cross-frame consistency.
- **Leave OIS on.** Optical stabilization helps. Aggressive *digital* (EIS)
  stabilization does not: it warps the effective intrinsics frame-to-frame,
  which the pipeline assumes are stable.

## Things that will produce garbage

- **Mirrors** — reconstructed as a real room behind the glass.
- **Windows** — the outdoor scene is fused as geometry beyond the wall. Close
  the curtains for a geometry-focused capture, or clip with `--max-depth`.
- **Glossy TVs and glass tabletops** — specular reflections give unstable depth.
- **People or pets moving through frame** — everything here assumes a static
  scene. A moving person is smeared across the reconstruction.
- **Multi-storey in one take** — the DXF is a single horizontal slice. Capture
  each floor as a separate run.

## Verifying before you spend GPU time

Run the frames stage alone and read the log:

```powershell
video2cad --video home.mp4 --workdir out --stages frames
```

A healthy capture keeps close to 100% of requested keyframes. If the log warns
that many were rejected for blur, re-shoot slower — that is much cheaper than
discovering it after the reconstruction pass.

## Getting an accuracy number

Before capturing, **tape-measure one wall** and note it. After the run, compare
against `extent_m` of the corresponding plane in `cad/planes.json`. That single
number tells you whether metric scale landed correctly, and is worth more than
any benchmark. Remember it only means anything with the nested checkpoint —
every other checkpoint is up-to-scale.
