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

NUM_DOCS=300000

CONFIGS=(
    # ═══ Chinchilla Scaling Law, C ≈ 5e15 FLOPs ═══
    "512  8  8  2048    33000000"   # N≈25.2M, D/N≈1.3
    "512  6  8  2048    44000000"   # N≈18.9M, D/N≈2.3
    "384  6  6  1536    79000000"   # N≈10.6M, D/N≈7.4
    "384  4  6  1536   118000000"   # N≈7.08M, D/N≈17
    "256  6  8  1024   177000000"   # N≈4.72M, D/N≈37
    "256  4  8  1024   265000000"   # N≈3.14M, D/N≈84
    "192  6  6  768    314000000"   # N≈2.65M, D/N≈119
)

# ── Run sweep ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Starting Chinchilla sweep: ${#CONFIGS[@]} configurations"
echo "  NUM_DOCS=${NUM_DOCS}"
echo "══════════════════════════════════════════════════"

RUN=1
for cfg in "${CONFIGS[@]}"; do
    read -r D_MODEL N_LAYERS N_HEADS D_FF TOKEN_BUDGET LR <<< "$cfg"

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Run ${RUN}/${#CONFIGS[@]}"
    echo "  d_model=${D_MODEL} | layers=${N_LAYERS} | heads=${N_HEADS} | d_ff=${D_FF}"
    echo "  token_budget=${TOKEN_BUDGET}"
    echo "──────────────────────────────────────────────────"

    uv run python main_gpu.py \
        --d_model       "$D_MODEL" \
        --num_layers    "$N_LAYERS" \
        --num_heads     "$N_HEADS" \
        --d_ff          "$D_FF" \
        --num_docs      "$NUM_DOCS" \
        --token_budget  "$TOKEN_BUDGET"

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