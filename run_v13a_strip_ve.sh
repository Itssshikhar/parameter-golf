#!/bin/bash
# v13a: Strip VE only — cleanest size fix, no TTT yet
#
# Change from v12: VE_ENABLED=0, TARGET_MB=15.2
# Everything else identical to v12 bankless.
#
# Purpose: establish submittable baseline under 16MB cap.
# Expected: ~1.093-1.100 sliding (VE strip costs 0.005-0.015 pre-quant),
#           size ~14.8 MB unpruned → no pruning needed.
set -e

LOGDIR="/workspace/parameter-golf"

export RUN_ID="v13a_strip_ve"
export SEED=1337
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096

# Architecture — same as v12 EXCEPT VE disabled
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export MATRIX_CLIP_SIGMAS=12.85
export PARTIAL_KEY_OFFSET=1
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export RECUR_LAYERS="4,5"
export RECUR_START_FRAC=0.5
export WARMDOWN_ITERS=4000

# v13a change: disable VE
export VE_ENABLED=0

# Fix: TARGET_MB in MiB, cap is 16,000,000 bytes decimal
# 15.2 MiB = 15,938,355 bytes → 61K headroom
export TARGET_MB=15.2

# No TTT — establish clean baseline first
export TTT_ENABLED=0

echo "========================================"
echo "  v13a: Strip VE (baseline, no TTT)"
echo "  VE_ENABLED=0, TARGET_MB=15.2"
echo "========================================"

rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_v13.py 2>&1 | tee "${LOGDIR}/v13a_strip_ve.log"
