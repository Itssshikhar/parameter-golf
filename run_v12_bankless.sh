#!/bin/bash
# v12: Bankless rewrite — per-layer CastedLinear, PR #1394-style Muon
#
# Core change: parameter banks (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
# replaced with standard per-layer CastedLinear weights. Muon now uses
# round-robin per-layer NS5 + flat all_reduce (matching PR #1394 exactly).
# GPTQ hooks directly on the training model — no separate _HessianGPT.
#
# Hypothesis: the bank architecture (3D tensors, joint optimization, unbank/rebank
# quantization pipeline) was the dominant cause of the ~0.08 BPB quantization gap
# in v9-v11. Per-layer weights quantize independently → no cross-layer error
# compounding through recurrence.
#
# Everything else kept from v9:
#   - SWA_WINDOW_SIZE=256, SWA_FULL_ATTN_LAYERS=5
#   - MATRIX_CLIP_SIGMAS=12.85 (competition SDClip k)
#   - RECUR_LAYERS=4,5, RECUR_START_FRAC=0.5
#   - BIGRAM_VOCAB_SIZE=3072, BIGRAM_DIM=112
#   - WARMDOWN_ITERS=4000
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v12_bankless"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture (same as v9)
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000

echo "========================================"
echo "  v12: Bankless Rewrite"
echo "  Per-layer CastedLinear + PR #1394 Muon"
echo "========================================"
echo "  Banks removed: qo_bank, kv_bank, mlp_up/down_bank"
echo "  Muon: round-robin NS5 + all_reduce (PR #1394)"
echo "  GPTQ: direct hooks on training model"
echo "  SWA_WINDOW_SIZE=256, RECUR_LAYERS=4,5"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v12_bankless.py 2>&1 | tee "${LOGDIR}/v12_bankless.log"
