# LLM-PIPELINE

A minimal from-scratch decoder-only transformer pipeline: BPE tokenizer → dataset → training → generation, followed by scaling-law experiments.

**Recent fixes:**
- Replaced naive pair search with `heapq` in `tokenizer.py` for faster BPE training.
- Switched `dataset.py` from stride-1 sliding window to non-overlapping chunks for efficient teacher forcing.

## Environment Setup

### Recommended: from lockfile
```bash
uv sync
```
This reads `pyproject.toml` + `uv.lock` and recreates the exact environment.

### Manual creation (how this environment was originally built)
```bash
uv init
uv add torch torchvision --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv add regex "datasets>=3.0" --index-strategy unsafe-best-match
```

## Run

```bash
uv run main.py
```

On a remote server (vast.ai, TPU, etc.):

```bash
chmod +x run_scaling_sweep.sh
./run_scaling_sweep.sh
```

Training logs, model checkpoints, and run configs are saved to `output/<timestamp>/`.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Training + evaluation + generation pipeline |
| `transformer.py` | Decoder-only transformer model |
| `tokenizer.py` | BPE tokenizer (train / encode / decode) |
| `dataset.py` | `LMDataset` for causal LM windowed samples |
| `experiment.py` | Experiment scripts |