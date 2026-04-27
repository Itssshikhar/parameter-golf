#!/bin/bash
# v10: Strip SWA + higher weight decay — isolate architectural impact on recurrence quant gap
#
# Hypothesis: SWA (window=256 on recurred layers 4,5) and lower weight decay (0.085)
# produce weight distributions that amplify quantization error through recurrence.
# Competition PRs (#1394, #1493) use full attention on all layers + WD 0.090-0.095.
#
# Changes from v9 (competition quant pipeline):
#   1. SWA_WINDOW_SIZE=0       — all 11 layers use full attention (was 256)
#   2. MUON_WD=0.095           — higher weight decay (was 0.085), matches competition
#
# Kept from v9:
#   - MATRIX_CLIP_SIGMAS=12.85 — competition SDClip k
#   - CALIB_MODE=train          — training data for Hessian calibration
#   - CALIB_RECURRENCE=1        — recurrence ON during calibration
#   - Embedding GPTQ            — via final_norm output hooks
#   - No pruning expected       — model should fit under 16MB natively
#
# Note: with SWA off, PARTIAL_KEY_OFFSET=1 applies to ALL 11 layers
# (was only layers 6-10 when SWA reserved 0-5 for windowed attention).
# This is a known confound — if gap closes, binary search can isolate.
#
# Expected impact:
#   - Fewer steps (~4500 vs 5400) due to full attention FLOPS
#   - Possibly worse pre-quant BPB (fewer steps + stronger regularization)
#   - If quant gap closes to ~0.01-0.02, SWA/WD was the culprit
#   - If gap stays at ~0.08, parameter banks are the bottleneck
set -e

LOGDIR="/workspace/parameter-golf"

# v10: no SWA + higher WD
export RUN_ID="v10_noswa_highwd"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# STRIPPED: sliding window attention disabled (all layers full attention)
export SWA_WINDOW_SIZE=0
export SWA_FULL_ATTN_LAYERS=0

# CHANGED: higher weight decay to match competition
export MUON_WD=0.095

# Kept from v9 / competition pipeline
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000

echo "========================================"
echo "  v10: No SWA + Higher WD (0.095)"
echo "  Testing if SWA/WD causes recurrence quant gap"
echo "========================================"
echo "  SWA_WINDOW_SIZE=0 (all full attention)"
echo "  MUON_WD=0.095 (was 0.085)"
echo "  MATRIX_CLIP_SIGMAS=12.85 (competition)"
echo "  RECUR_LAYERS=4,5"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGDIR}/v10_noswa_highwd.log"
