#!/bin/bash
# v13c: TTT on top of the winning v13a/b config
#
# Uses whichever strip (VE or bigram) won from v13a/b.
# Default: strip VE (edit if v13b wins).
#
# Purpose: test if TTT works on bankless architecture.
# CRITICAL: TTT failed catastrophically on bank-based code (v4: +0.089, v5: +0.046).
# This is the first test on bankless. If it fails, TTT is dead for our stack.
#
# TTT parameter filtering: only trains large weight matrices (>65536 elements),
# excludes embeddings, scales, and control tensors.
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13c_ttt"
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

# v13c change: enable TTT with conservative settings
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_MOMENTUM=0.9

echo "========================================"
echo "  v13c: TTT on bankless (first test)"
echo "  TTT_ENABLED=1, VE stripped"
echo "  TTT params filtered to large matrices only"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13c_ttt.log"
