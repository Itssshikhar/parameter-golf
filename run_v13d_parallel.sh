#!/bin/bash
# v13d: Add parallel residuals to winning config
#
# Change: PARALLEL_START_LAYER=7 (layers 7-10 run attn+MLP in parallel)
# Uses winning strip from v13a/b + TTT if v13c works.
#
# Purpose: test if parallel residuals help our stack.
# Note: if VE is on, parallel at layer 9-10 reduces VE effectiveness
# (MLP won't see VE-enriched attention output in parallel mode).
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13d_parallel"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — winner from v13a/b (default: strip VE)
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000
export VE_ENABLED=0
export TARGET_MB=15.2

# v13d change: parallel residuals from layer 7
export PARALLEL_START_LAYER=7

# TTT — set based on v13c results (default: off until proven)
export TTT_ENABLED=0

echo "========================================"
echo "  v13d: Parallel residuals from L7"
echo "  PARALLEL_START_LAYER=7"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13d_parallel.log"
