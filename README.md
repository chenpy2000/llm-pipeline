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
uv run main_local.py
```

Run all sweep experiments locally:

```bash
chmod +x run_local.sh
./run_local.sh
```

On a remote GPU server (requires a PyTorch base template, e.g. vast.ai [PyTorch (Vast)](https://hub.docker.com/r/vastai/pytorch/)):

```bash
chmod +x run_gpu.sh
./run_gpu.sh
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