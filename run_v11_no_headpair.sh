#!/bin/bash
# v11: Disable paired-head Muon — test if bank optimizer is the quant gap culprit
#
# Hypothesis: Paired-head NS5 on QK/KV banks creates block-orthogonal weight
# structure at the head-pair level. INT6 quantization breaks these correlations,
# and the error compounds 900x through recurrence. Standard per-slice NS5
# (closer to competition's MuonEq-R) may produce more quantization-robust weights.
#
# Changes from v9 (competition quant pipeline):
#   1. HEAD_PAIR_NS=0  — disable paired-head Newton-Schulz on QK/KV banks
#                        Muon applies standard NS5 per bank slice instead
#
# Everything else matches v9:
#   - SWA_WINDOW_SIZE=256, SWA_FULL_ATTN_LAYERS=5 (keep SWA — v10 ruled it out)
#   - MUON_WD=0.085 (default — v10 ruled out higher WD)
#   - MATRIX_CLIP_SIGMAS=12.85 (competition SDClip k)
#   - Competition quant pipeline (train data, embedding GPTQ, no pruning)
#
# If gap closes → paired-head Muon was the culprit
# If gap stays → the bank architecture itself (3D storage + joint optimization) is the issue
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v11_no_headpair"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# CHANGED: disable paired-head Newton-Schulz
export HEAD_PAIR_NS=0

# Same as v9 (competition quant pipeline)
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
echo "  v11: No Paired-Head Muon"
echo "  Testing if head-pair NS5 causes recurrence quant gap"
echo "========================================"
echo "  HEAD_PAIR_NS=0 (standard per-slice NS5)"
echo "  SWA_WINDOW_SIZE=256 (kept — v10 ruled out)"
echo "  MUON_WD=0.085 (default — v10 ruled out higher)"
echo "  MATRIX_CLIP_SIGMAS=12.85 (competition)"
echo "  RECUR_LAYERS=4,5"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGDIR}/v11_no_headpair.log"
