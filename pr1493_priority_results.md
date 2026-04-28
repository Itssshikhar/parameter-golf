# PR1493 Priority Experiments — Results Log

Comparator from `pr1493_priority_experiments.md`:

```text
PR1493 reproduced, seed 42, QK_GAIN_INIT=5.25:
  quantized_ttt: 1.08103358

Best local TTT sweep (TTT_LR=0.007, TTT_EPOCHS=5):
  quantized_ttt: 1.08079274
```

To matter for leaderboard acceptance margin, we likely need another `~0.0017–0.0020 BPB`,
not noise-level movement. Anything that does not beat ~`1.08079` on this seed is not a
real win.

## Environment

- 8x NVIDIA H100 80GB HBM3
- torch 2.9.1+cu128, flash_attn_3 cu128_torch291
- Dataset: `kevclark/parameter-golf` SP8192, 128 train shards, 1 val shard
- Common env: `SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5`
- Wallclock cap: 600s training + GPTQ + post-train eval (~15-20 min total per run)

## Results table

Filled in as each experiment finishes. `pre` = pre-quantization post-EMA val_bpb,
`q` = quantized val_bpb, `q_sw` = quantized_sliding_window val_bpb,
`q_ttt` = quantized_ttt val_bpb (primary metric), `size` = Total submission size bytes,
`stop_step` = step reached at wallclock cap.

| # | Experiment | flag(s) | pre | q | q_sw | q_ttt | size | stop_step | status |
|---|------------|---------|-----|---|------|-------|------|-----------|--------|
| 0 | baseline_ttt | — | — | — | — | — | — | — | aborted (user requested skip after warmup) |
| 1 | docshuffle | `DOC_SHUFFLE_ENABLED=1` | — | — | — | — | — | — | running |
| 2 | wd | `WD_SCHEDULE_ENABLED=1` | — | — | — | — | — | — | queued |
| 3 | iha | `IHA_ENABLED=1` | — | — | — | — | — | — | queued |
| 4 | mtp | `MTP_WEIGHT=0.10 MTP_STEPS=1` | — | — | — | — | — | — | queued |
| 5 | evalloop3 | `EVAL_NUM_LOOPS=3` | — | — | — | — | — | — | queued |

## Per-experiment notes

### baseline_ttt (aborted)

Started successfully but stopped by user after warmup completed (20/20 + loop_warmup 20/20).
Not run to completion. Machine parity vs. the runbook's expected shape
(`pre ≈ 1.0875–1.0880`, `q_sw ≈ 1.083`, `q_ttt ≈ 1.08079–1.08103`) is therefore unverified.

### docshuffle (running)

`DOC_SHUFFLE_ENABLED=1` activates `DocumentShuffleLoader` instead of
`ShuffledSequenceLoader`. The loader logs:

```
doc_shuffle:bos=1 files=16 docs=1929107
```

(per rank — 16 shards × 8 ranks = 128 train shards total, ~15.4M docs across all ranks).
Training proceeding normally; first 3k steps land in the same train_loss range as the
baseline trajectory. Tokens/sec dropped from ~7.6M/s (steps 500-1500) to ~6.6M/s
(step 3000) — the document loader is doing more index work per batch.

## Errors / learnings (live)

- _none yet_

## How this file is updated

A scheduled cron (`*/10` cadence offset to `7,17,27,37,47,57 * * * *`) re-runs the
documentation prompt every 10 minutes, parses any newly-written experiment logs, and
pushes to the `shikhar` branch. Cron job ID `1aa86940`. Stops on session close or via
`CronDelete 1aa86940`.
