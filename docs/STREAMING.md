# Streaming mode (DA3-Streaming)

`run_streaming.py` uses the upstream **DA3-Streaming** implementation for long or
whole-home captures: overlapping chunks aligned with Sim(3), plus optional loop
closure across revisited places. It is ByteDance's own pipeline, built on
VGGT-Long.

## Why it isn't committed to this repo

The `Depth-Anything-3/` streaming tree is **git-ignored**, for two reasons:

1. It is large.
2. Its loop-closure component, **SALAD, is GPL-3.0** — shipping it inside this
   Apache-2.0 repo would create a copyleft conflict. See [LICENSING.md](LICENSING.md).

So a fresh clone fetches it once.

## One-time setup

```bash
# 1. fetch DA3-Streaming WITH submodules, next to this repo or anywhere
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3

# 2. point video2cad at its da3_streaming/ folder
#    (or place it at ./Depth-Anything-3/da3_streaming, the default search path)
export DA3_STREAMING_DIR=/abs/path/to/Depth-Anything-3/da3_streaming   # Linux/mac
#   set DA3_STREAMING_DIR=C:\path\to\Depth-Anything-3\da3_streaming     # Windows

# 3. install streaming deps + its weights
uv pip install --python .venv/bin/python -e '.[streaming]'
cd "$DA3_STREAMING_DIR" && bash scripts/download_weights.sh && cd -

# 4. run
python run_streaming.py home.mp4 out_stream \
  --target-frames 300 --chunk-size 10 --process-res 336 --no-loop --rescale-height 2.1
```

If you keep DA3-Streaming at `./Depth-Anything-3/da3_streaming`, you can skip
`DA3_STREAMING_DIR`; `run_streaming.py` finds it there automatically. Pass
`--streaming-dir` to override per run.

## Loop closure

On by default. Detection uses a DINOv2-based SALAD descriptor per keyframe,
retrieves revisited places by cosine similarity above 0.85, verifies
geometrically, and corrects the trajectory with a Sim(3) pose-graph optimizer.

- Use `--no-loop` unless the weight download supplied `dino_salad.ckpt`. It drops
  the SALAD (GPL) requirement entirely and never imports it.
- Loop closure only helps when the capture **actually revisits** a place (walk
  through rooms and return). For a single-room or open scan it does nothing
  useful, so `--no-loop` is the right default there.

## Choosing chunk size for your GPU

Peak VRAM is set by chunk size and resolution. A **10-frame chunk at
`--process-res 336`** is the conservative starting point for a 16 GB GPU. Increase
either setting only when it fits, and reduce them further if CUDA runs out of
memory.

## Metric scale

Same rule as every path: only the NESTED checkpoint emits true meters. With the
streaming default weights the cloud is up-to-scale, so pass `--rescale-height`
(door ≈ 2.1 m, ceiling ≈ 2.7 m) to get the DXF into meters.
