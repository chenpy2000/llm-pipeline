#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Scaling Law Sweep — L(N) experiment
# Measures converged val PPL as a function of model parameters.
#
# Usage:
#   chmod +x run_scaling_sweep.sh
#   ./run_scaling_sweep.sh
#
# Works on: vast.ai (CUDA), Google TPU (via PyTorch/XLA), local GPU
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment setup ─────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════"
echo "  Scaling Law Sweep — Environment Setup"
echo "══════════════════════════════════════════════════"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing dependencies ..."
uv sync

# ── Sweep configuration ──────────────────────────────────────────────────────
# Fixed across all runs:
#   - NUM_DOCS=150000  (~200M tokens, enough for biggest model)
#   - context_length=256, vocab_size=4000  (set in main.py defaults)
#   - early_stop=5  (train to convergence)
#   - eval_interval=1000  (set in main.py default)

NUM_DOCS=150000
EARLY_STOP=5

# Model configs: d_model | num_layers | num_heads | d_ff
# d_ff = 4 * d_model throughout
CONFIGS=(
    "64   2  2  256"
    "128  4  4  512"
    "192  4  6  768"
    "256  6  8  1024"
    "384  6  6  1536"
    "512  8  8  2048"
)

# ── Run sweep ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Starting sweep: ${#CONFIGS[@]} configurations"
echo "  NUM_DOCS=${NUM_DOCS} | early_stop=${EARLY_STOP}"
echo "══════════════════════════════════════════════════"

RUN=1
for cfg in "${CONFIGS[@]}"; do
    read -r D_MODEL N_LAYERS N_HEADS D_FF <<< "$cfg"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Run ${RUN}/${#CONFIGS[@]}"
    echo "  d_model=${D_MODEL} | layers=${N_LAYERS} | heads=${N_HEADS} | d_ff=${D_FF}"
    echo "──────────────────────────────────────────────────"

    uv run python main_local.py \
        --d_model    "$D_MODEL" \
        --num_layers "$N_LAYERS" \
        --num_heads  "$N_HEADS" \
        --d_ff       "$D_FF" \
        --num_docs   "$NUM_DOCS" \
        --early_stop "$EARLY_STOP"

    echo "  ✓ Run ${RUN} complete"
    RUN=$((RUN + 1))
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Sweep complete! ${#CONFIGS[@]} runs finished."
echo "  Results in: output/"
echo "══════════════════════════════════════════════════"
echo ""
echo "Next: collect final_val_ppl and total_params from each"
echo "  output/*/run_config_*.json"
echo "  and plot log(params) vs log(val_ppl)"
