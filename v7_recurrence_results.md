# v7 Recurrence Quantization Fix Results

**Date:** 2026-04-26/27  
**Branch:** shikhar  
**Baseline:** v6 (paired-head Muon, no recurrence) — sliding BPB 1.1029

## Objective

Test two approaches to fix depth recurrence's quantization problem (layers 4,5 replayed twice cause INT6 quantization error compounding):

1. **INT8 for recurred layers** (`RECUR_INT8=1`): Quantize layers 4,5 at INT8 (127 clip range) instead of INT6 (31 clip range)
2. **Skip recurrence at eval** (`RECUR_SKIP_EVAL=1`): Train with 13 virtual layers (recurrence active), evaluate with 11 physical layers (recurrence disabled)

## Configuration

All runs share:
- Seed: 1337
- `RECUR_LAYERS="4,5"` (depth recurrence on layers 4,5, activated at 50% wallclock ~300s)
- Paired-head Muon enabled
- `TRAIN_SEQ_LEN=4096`, `EVAL_SEQ_LEN=4096`
- `SWA_WINDOW_SIZE=256`, `SWA_FULL_ATTN_LAYERS=5`
- `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=112`
- `WARMDOWN_ITERS=4000`
- GPTQ with autoregressive calibration (64 seqs x 2048 tokens, temp=0.8)
- Target size: 15.9MB
- Sliding window eval stride: 64

## Results Summary

| Run | Config | Pre-quant BPB | Sliding BPB | Quant Gap | Pruning | File Size | Verdict |
|-----|--------|--------------|-------------|-----------|---------|-----------|---------|
| **v6** | No recurrence (baseline) | 1.0986 | **1.1029** | 0.0043 | ~2% | ~15.5MB | **Best** |
| **v7a** | INT8 recurred layers | 1.0888 | 1.2263 | 0.1375 | 9.3% (2.4M/26M) | 16.54MB | Failed |
| **v7b** | Skip recurrence at eval | 1.0874 | 1.1961 | 0.1087 | None needed | 16.17MB | Failed |
| **v7c** | INT8 + skip recurrence | 1.0886 | 1.2251 | 0.1365 | 9.2% (2.4M/26M) | 16.54MB | Failed |

## Detailed Run Metrics

### v7a — INT8 for Recurred Layers (`RECUR_INT8=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5212/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 115.14ms |
| Training | Recurrence activated | Step 2935 @ 300s |
| Training | SWA start | Step 4450 |
| Training | Late QAT enabled | Step 4628 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1325 |
| Training | Final val BPB (step 5212) | 1.0898 |
| Post-EMA | Pre-quant BPB | **1.0888** |
| Quantization | INT8 override | Layers {4, 5} |
| Quantization | Unpruned size | 16.63MB |
| Quantization | Pruning | 2,423,219/25,952,256 weights (9.3%) |
| Quantization | Final compressed size | 16,538,546 bytes |
| Quantization | Total submission size | 16,672,225 bytes |
| Eval | Roundtrip BPB | 1.2339 |
| Eval | **Sliding window BPB** | **1.2263** |
| Eval | Sliding eval time | 151,961ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3141      -         51s
1000     3.0185      -         102s
2000     3.0814      -         204s
2935     [recurrence activated]  300s
3000     2.9144      -         342s
4000     2.9303      1.1325    458s
5000     2.7795      -         575s
5212     -           1.0898    600s (stopped)
```

### v7b — Skip Recurrence at Eval (`RECUR_SKIP_EVAL=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5406/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 111.00ms |
| Training | Recurrence activated | Step 2929 @ 300s |
| Training | SWA start | Step 4700 |
| Training | Late QAT enabled | Step 4844 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1388 |
| Training | Final val BPB (step 5406) | 1.0886 |
| Post-EMA | Pre-quant BPB | **1.0874** |
| Quantization | All INT6 (no override) | - |
| Quantization | Unpruned size | 15.55MB |
| Quantization | Pruning | **None needed** (already fits) |
| Quantization | Final compressed size | 16,169,729 bytes |
| Quantization | Total submission size | 16,303,408 bytes |
| Eval | Recurrence disabled | Yes (`recur_skip_eval=1`) |
| Eval | Roundtrip BPB | 1.2022 |
| Eval | **Sliding window BPB** | **1.1961** |
| Eval | Sliding eval time | 126,725ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3067      -         51s
1000     3.0239      -         102s
2000     3.0806      -         205s
2929     [recurrence activated]  300s
3000     2.9236      -         320s
4000     2.9416      1.1388    436s
5000     2.7964      -         552s
5406     -           1.0886    600s (stopped)
```

### v7c — Both INT8 + Skip Recurrence (`RECUR_INT8=1, RECUR_SKIP_EVAL=1`)

| Phase | Metric | Value |
|-------|--------|-------|
| Training | Steps completed | 5410/20000 |
| Training | Wallclock | 600s (10min cap) |
| Training | Step avg | 110.92ms |
| Training | Recurrence activated | Step 2930 @ 300s |
| Training | SWA start | Step 4700 |
| Training | Late QAT enabled | Step 4848 (scale 0.15) |
| Training | Mid-train val BPB (step 4000) | 1.1398 |
| Training | Final val BPB (step 5410) | 1.0897 |
| Post-EMA | Pre-quant BPB | **1.0886** |
| Quantization | INT8 override | Layers {4, 5} |
| Quantization | Unpruned size | 16.63MB |
| Quantization | Pruning | 2,394,896/25,952,256 weights (9.2%) |
| Quantization | Final compressed size | 16,538,571 bytes |
| Quantization | Total submission size | 16,672,250 bytes |
| Eval | Recurrence disabled | Yes (`recur_skip_eval=1`) |
| Eval | Roundtrip BPB | 1.2332 |
| Eval | **Sliding window BPB** | **1.2251** |
| Eval | Sliding eval time | 126,953ms |

Training curve:
```
step     train_loss  val_bpb   time
0        -           3.4832    0s
500      3.3032      -         51s
1000     3.0218      -         102s
2000     3.0853      -         205s
2930     [recurrence activated]  300s
3000     2.9247      -         319s
4000     2.9484      1.1398    435s
5000     2.7953      -         552s
5410     -           1.0897    600s (stopped)
```

## Analysis

### Recurrence helps pre-quant but destroys post-quant

All three recurrence runs achieved better pre-quant BPB (~1.0874-1.0888) vs v6 (1.0986), confirming recurrence genuinely improves the model by ~0.01 BPB. However, no tested approach could preserve this gain through quantization.

### Why each approach failed

**v7a (INT8 recurred layers):** INT8 uses 8 bits per weight vs INT6's 6 bits. For layers 4,5 (6 weight matrices each = 12 matrices total), this increased total model size from ~15.5MB to ~16.6MB. To fit the 15.9MB budget, 9.3% of all weights had to be pruned (zeroed out), destroying far more quality than the extra 2 bits of precision saved. The quant gap exploded to 0.1375 BPB.

**v7b (Skip recurrence at eval):** The model trained with 13 virtual layers (layers 4,5 replayed) but was evaluated with only 11 physical layers. The model learned features that depend on the second pass through layers 4,5 — removing recurrence at eval is equivalent to removing 2 layers from a trained network. The 0.1087 BPB gap shows the model cannot simply ignore the recurrence it was trained with. Notably, this had no pruning penalty (15.55MB fits under 15.9MB), so the entire gap is from the architectural mismatch.

**v7c (Both combined):** Worst of both worlds. INT8's size penalty forced heavy pruning (9.2%), AND the model lost its recurrence at eval. The results are comparable to v7a alone (1.2251 vs 1.2263), confirming the skip-eval adds negligible benefit on top of the pruning damage.

### Key insight: size budget is the binding constraint

The 15.9MB target is extremely tight. Any approach that increases per-weight bit count (INT8 vs INT6) requires proportionally more pruning, which overwhelms any precision benefit. The only way to use higher-precision quantization for recurred layers would be to:
- Increase the size budget (not allowed by competition rules)
- Reduce model size elsewhere to compensate (e.g., fewer layers or smaller dimensions)
- Find a quantization method that doesn't increase storage size

## Historical Run Comparison

| Run | Description | Pre-quant BPB | Sliding BPB | Quant Gap |
|-----|-------------|--------------|-------------|-----------|
| v1 | Baseline (no recurrence) | 1.0985 | 1.1049 | 0.0064 |
| v3 | Recurrence (unmodified INT6) | 1.0949 | 1.1792 | 0.0843 |
| v4 | TTT (redundant, all-ranks) | 1.0988 | 1.1039 / TTT:1.1932 | - |
| v5 | TTT (distributed fix) | ~same | 1.1041 / TTT:1.1497 | - |
| **v6** | **Paired-head Muon, no recur** | **1.0986** | **1.1029** | **0.0043** |
| v7a | Recur + INT8 recurred layers | 1.0888 | 1.2263 | 0.1375 |
| v7b | Recur + skip recur at eval | 1.0874 | 1.1961 | 0.1087 |
| v7c | Recur + INT8 + skip eval | 1.0886 | 1.2251 | 0.1365 |

## Conclusion

Depth recurrence on layers 4,5 is incompatible with the current INT6 quantization + 15.9MB size budget. Neither INT8 precision, nor skipping recurrence at eval, nor both combined can close the gap. **v6 (no recurrence, paired-head Muon) remains our best at 1.1029 BPB.**

Future directions should focus on improving pre-quant BPB through approaches that don't create quantization-hostile architectures, or on reducing the quantization gap for the existing architecture.

## Artifacts

- Logs: `logs/v7a_recur_int8.txt`, `logs/v7b_recur_skipeval.txt`, `logs/v7c_recur_both.txt`
- Console logs: `v7a_recur_int8.log`, `v7b_recur_skipeval.log`, `v7c_recur_both.log`
- Run script: `run_v7_recur_tests.sh`
- HuggingFace: `shikhar007/parameter-golf-gram-ns/models/` and `logs/`
