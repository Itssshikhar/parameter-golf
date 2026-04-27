#!/bin/bash
# v7 recurrence quantization fix tests
# v7a: INT8 for recurred layers only
# v7b: Skip recurrence at eval only
# v7c: Both INT8 + skip recurrence at eval
set -e

LOGDIR="/workspace/parameter-golf"

# Common settings matching our best config
export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export SEED=1337

# Enable recurrence on layers 4,5
export RECUR_LAYERS="4,5"

echo "========================================"
echo "  v7a: recurrence + INT8 for recurred layers"
echo "========================================"
export RUN_ID="v7a_recur_int8"
export RECUR_INT8=1
export RECUR_SKIP_EVAL=0
rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGDIR}/v7a_recur_int8.log"

echo "========================================"
echo "  v7b: recurrence + skip recurrence at eval"
echo "========================================"
export RUN_ID="v7b_recur_skipeval"
export RECUR_INT8=0
export RECUR_SKIP_EVAL=1
rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGDIR}/v7b_recur_skipeval.log"

echo "========================================"
echo "  v7c: recurrence + INT8 + skip recurrence at eval"
echo "========================================"
export RUN_ID="v7c_recur_both"
export RECUR_INT8=1
export RECUR_SKIP_EVAL=1
rm -rf ~/.cache/torch_extensions 2>/dev/null || true
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGDIR}/v7c_recur_both.log"

echo ""
echo "========================================"
echo "  v7 SUMMARY"
echo "========================================"
for RUN in v7a_recur_int8 v7b_recur_skipeval v7c_recur_both; do
    LOGFILE="${LOGDIR}/${RUN}.log"
    PRE=$(grep "final_prequant_exact" "${LOGFILE}" 2>/dev/null | grep "val_bpb" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/')
    SLIDING=$(grep "final_int6_sliding_window_exact" "${LOGFILE}" 2>/dev/null | grep "val_bpb" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/')
    echo "  ${RUN}: pre-quant=${PRE:-N/A} sliding=${SLIDING:-N/A}"
done
echo "  v6 baseline (no recurrence):  sliding=1.1029"
