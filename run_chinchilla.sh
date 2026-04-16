#!/usr/bin/env bash
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════════
# Chinchilla IsoFLOP Sweep — L(N, D) at fixed compute C = 6·N·D
# For each compute budget C, sweep N while holding C constant.
# Minimum val PPL across N at fixed C gives the compute-optimal model size.
#
# Usage:
#   chmod +x run_chinchilla.sh
#   ./run_chinchilla.sh
# ══════════════════════════════════════════════════════════════════════════════

# ── Environment setup ─────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════"
echo "  Chinchilla IsoFLOP Sweep — Environment Setup"
echo "══════════════════════════════════════════════════"

if ! command -v uv &> /dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies ..."

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "5090|5080|B100|B200"; then
    echo "Blackwell+ GPU detected — installing PyTorch with cu128 ..."
    uv pip install torch torchvision --index https://download.pytorch.org/whl/cu128 --system --break-system-packages --index-strategy unsafe-best-match
else
    echo "Pre-Blackwell GPU detected — using template PyTorch"
fi

uv pip install regex "datasets>=3.0" --system --break-system-packages --index-strategy unsafe-best-match

# ── Sweep configuration ──────────────────────────────────────────────────────
# Each row: d_model | n_layers | n_heads | d_ff | token_budget
# Grouped by compute budget C ≈ 6 · N · D.
#
# IMPORTANT: token_budget must not exceed train tokens (~180M at NUM_DOCS=150000).
# If it does, the single-epoch loop will exit early and the run will be invalid.
# All budgets below stay under 150M for safety.

NUM_DOCS=150000

CONFIGS=(
    # ═══ Compute ≈ 3e14 FLOPs ═══
    "64   2  2  256    140000000"   # ~350K params × 140M tokens
    "128  4  4  512    38000000"    # ~1.3M × 38M
    "192  4  6  768    20000000"    # ~2.5M × 20M
    "256  6  8  1024   9000000"     # ~5.7M × 9M
    "384  6  6  1536   4000000"     # ~12M  × 4M

    # ═══ Compute ≈ 1e15 FLOPs ═══
    "128  4  4  512    128000000"   # ~1.3M × 128M
    "192  4  6  768    67000000"    # ~2.5M × 67M
    "256  6  8  1024   29000000"    # ~5.7M × 29M
    "384  6  6  1536   14000000"    # ~12M  × 14M
    "512  8  8  2048   6000000"     # ~27M  × 6M

    # ═══ Compute ≈ 2e15 FLOPs ═══
    "192  4  6  768    133000000"   # ~2.5M × 133M
    "256  6  8  1024   58000000"    # ~5.7M × 58M
    "384  6  6  1536   28000000"    # ~12M  × 28M
    "512  8  8  2048   12000000"    # ~27M  × 12M
)

# ── Run sweep ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Starting Chinchilla sweep: ${#CONFIGS[@]} configurations"
echo "  NUM_DOCS=${NUM_DOCS} | early_stop=0 (default, required for Chinchilla)"
echo "══════════════════════════════════════════════════"

RUN=1
for cfg in "${CONFIGS[@]}"; do
    read -r D_MODEL N_LAYERS N_HEADS D_FF TOKEN_BUDGET <<< "$cfg"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Run ${RUN}/${#CONFIGS[@]}"
    echo "  d_model=${D_MODEL} | layers=${N_LAYERS} | heads=${N_HEADS} | d_ff=${D_FF}"
    echo "  token_budget=${TOKEN_BUDGET}"
    echo "──────────────────────────────────────────────────"

    uv run python main_gpu.py \
        --d_model      "$D_MODEL" \
        --num_layers   "$N_LAYERS" \
        --num_heads    "$N_HEADS" \
        --d_ff         "$D_FF" \
        --num_docs     "$NUM_DOCS" \
        --token_budget "$TOKEN_BUDGET"

    echo "  ✓ Run ${RUN} complete"
    RUN=$((RUN + 1))
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Chinchilla sweep complete! ${#CONFIGS[@]} runs finished."
echo "  Results in: output/"
echo "══════════════════════════════════════════════════"
echo ""
echo "Next: for each compute budget, plot log(val_ppl) vs log(total_params)"
echo "  → expect a U-shape at each fixed C"
echo "  → locus of U-minima across C gives the Chinchilla exponents"
echo "  → collect from output/*/run_config_*.json (total_params, final_val_ppl, tokens_seen)"