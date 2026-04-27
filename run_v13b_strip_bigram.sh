#!/bin/bash
# v13b: Strip bigram instead of VE — compare which strip costs less BPB
#
# Change from v12: BIGRAM_VOCAB_SIZE=0, TARGET_MB=15.2
# VE stays on. Will need pruning (~0.15 MB over after strip).
#
# Purpose: head-to-head vs v13a. If bigram costs less than VE, use this.
# Expected: ~1.093-1.103 sliding, size ~15.35 MB unpruned → light pruning.
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13b_strip_bigram"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — same as v12 EXCEPT bigram disabled
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000

# v13b change: disable bigram
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=112

# VE stays on
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# Fix: correct TARGET_MB
export TARGET_MB=15.2

# No TTT
export TTT_ENABLED=0

echo "========================================"
echo "  v13b: Strip Bigram (VE on, no TTT)"
echo "  BIGRAM_VOCAB_SIZE=0, TARGET_MB=15.2"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13b_strip_bigram.log"
