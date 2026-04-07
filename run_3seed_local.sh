#!/bin/bash
# 3-seed validation of best SWA config: seq4096, w=256, 5 full attn layers
# Run locally on 8xH100

set -e

SEEDS=(1337 42 7)
LOGDIR="/workspace/parameter-golf"

export TRAIN_SEQ_LEN=4096
export EVAL_SEQ_LEN=4096
export SWA_WINDOW_SIZE=256
export SWA_FULL_ATTN_LAYERS=5
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000

for SEED_VAL in "${SEEDS[@]}"; do
    export SEED=$SEED_VAL
    export RUN_ID="3seed_s${SEED_VAL}"
    LOGFILE="${LOGDIR}/${RUN_ID}.log"

    echo "========================================"
    echo "  Starting SEED=${SEED_VAL} -> ${LOGFILE}"
    echo "========================================"

    # Clear torch compile cache between runs
    rm -rf ~/.cache/torch_extensions 2>/dev/null || true

    torchrun --standalone --nproc_per_node=8 train_gpt_swa.py 2>&1 | tee "${LOGFILE}"

    echo "========================================"
    echo "  SEED=${SEED_VAL} DONE"
    echo "========================================"
done

# Summary
echo ""
echo "========================================"
echo "  3-SEED SUMMARY"
echo "========================================"
BPBS=()
for SEED_VAL in "${SEEDS[@]}"; do
    LOGFILE="${LOGDIR}/3seed_s${SEED_VAL}.log"
    SLIDING=$(grep "final_int6_sliding_window_exact" "${LOGFILE}" | grep "val_bpb" | head -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/')
    echo "  Seed ${SEED_VAL}: sliding BPB = ${SLIDING}"
    BPBS+=("${SLIDING}")
done

# Calculate mean with python
python3 -c "
bpbs = [float(x) for x in '${BPBS[0]} ${BPBS[1]} ${BPBS[2]}'.split()]
mean = sum(bpbs) / len(bpbs)
print(f'  Mean sliding BPB: {mean:.4f}')
print(f'  Current #1:       1.1147')
print(f'  Delta:            {mean - 1.1147:+.4f}')
"
