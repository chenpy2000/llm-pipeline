# llm-pipeline

A from-scratch project for training a coding base LLM: tokenizer, dataset pipeline, decoder-only transformer, pre-training loop, generation, and the post-training path toward a coding agent.

This repository is now moving from broad pre-training experiments into post-training. The scaling-law work is preserved as the research foundation; the active next phase is to make the base model more useful for coding tasks.

## 1. Research for Pre-Training

The first phase studied how to spend pre-training compute for a small from-scratch language model. The project replicated a Chinchilla-style scaling-law experiment across several model sizes and token budgets, then used the result to choose a better direction for base-model training.

![Corrected Chinchilla scaling law result](chinchilla_curve.png)

The main pre-training result: at this small scale, the compute-optimal data-to-parameter ratio was much higher than the classical large-model Chinchilla rule of `D/N ~= 20`, and decreased as compute increased.

| Compute | Optimal N | Optimal D/N |
|---|---:|---:|
| `5e14` FLOPs | about `0.44M` | about `430` |
| `1e15` FLOPs | about `0.79M` | about `270` |
| `5e15` FLOPs | about `2.65M` | about `120` |

For the detailed experiment record, including the original README, sweep scripts, plots, and notes, see the [`1_scaling_law` branch](https://github.com/chenpy2000/llm-pipeline/tree/1_scaling_law).

## 2. Post-Training

The next phase starts here. The goal is to move from a general pre-trained base model toward a coding-oriented assistant model.

Post-training work will focus on:

- settling the model architecture before running more training;
- keeping future pre-training runs smaller and more targeted;
- adding supervised fine-tuning data for coding tasks;
- adding evaluation for code generation, editing, and instruction following;
- exploring preference tuning or other alignment steps;
- building toward an agent loop that can read code, modify files, run tests, and improve from feedback.

## Active Code

| Path | Purpose |
|---|---|
| `main.py` | Main pre-training, evaluation, checkpointing, and sample generation entry point |
| `transformer.py` | Decoder-only transformer model |
| `tokenizer.py` | BPE tokenizer implementation |
| `dataset.py` | Causal language-modeling dataset |
| `tokenizer/` | Saved tokenizer artifacts |
| `chinchilla_curve.png` | Final pre-training scaling-law summary image |
| `archive/` | Archived local scripts, sweep runners, extra plots, and notes from the scaling-law phase |

## Setup

Install from the lockfile:

```bash
uv sync
```

Run the active training pipeline:

```bash
uv run main.py
```

Useful overrides:

```bash
uv run main.py --d_model 128 --num_layers 4 --num_heads 4 --d_ff 512
uv run main.py --num_docs 500000 --token_budget 20000000 --learning_rate 0.0012
```

Training logs, checkpoints, and run configs are written to `output/<timestamp>/`.
