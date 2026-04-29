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
| 1 | docshuffle | `DOC_SHUFFLE_ENABLED=1` | 1.09005 | 1.10121 | 1.08448 | **1.08279** | 16,033,898 | 4526/20000 | done |
| 2 | wd | `WD_SCHEDULE_ENABLED=1` | — | — | — | — | — | — | running |
| 3 | iha | `IHA_ENABLED=1` | — | — | — | — | — | — | queued |
| 4 | mtp | `MTP_WEIGHT=0.10 MTP_STEPS=1` | — | — | — | — | — | — | queued |
| 5 | evalloop3 | `EVAL_NUM_LOOPS=3` | — | — | — | — | — | — | queued |

## Per-experiment notes

### baseline_ttt (aborted)

Started successfully but stopped by user after warmup completed (20/20 + loop_warmup 20/20).
Not run to completion. Machine parity vs. the runbook's expected shape
(`pre ≈ 1.0875–1.0880`, `q_sw ≈ 1.083`, `q_ttt ≈ 1.08079–1.08103`) is therefore unverified.

### docshuffle (done — clear regression)

`DOC_SHUFFLE_ENABLED=1` activates `DocumentShuffleLoader` instead of
`ShuffledSequenceLoader`. Loader log: `doc_shuffle:bos=1 files=16 docs=1929107`
per rank (16 shards × 8 ranks = 128 train shards total, ~15.4M docs total).

**Result: q_ttt = 1.08279 vs comparator 1.08079 → Δ = +0.00200 BPB (worse).**
Not noise — that's roughly the same magnitude as the gap we'd need to *win*, but in the
wrong direction. Reasons it likely hurt:

- Tokens/sec dropped from ~7.6M/s (steps 500-1500) to ~6.6M/s by step 3000 and ~6.0M/s
  by step 4500. Wallclock cap is fixed at 588s post-reserve, so the doc loader's per-batch
  overhead cost ~10% of total training steps. docshuffle stopped at step 4526/20000;
  baseline's expected stop is around 4900-5000.
- Document boundaries make many short windows that span a single doc, so the model sees
  fewer cross-doc contexts per step. At this training budget that's net-negative.

Submission size **16,033,898 bytes — over the 16M limit by 33,898 bytes**. Even if it
*had* been better, it wouldn't be submittable without code-size minification.

**Verdict: drop.**

## Errors / learnings (live)

- _none yet_

## How this file is updated

A scheduled cron (`7,17,27,37,47,57 * * * *`) re-runs the documentation prompt every
10 minutes, parses any newly-finished experiment logs (gated by the orchestrator's
`=== [N/5] <name> done at` line), and pushes to the `shikhar` branch only when the
markdown actually changed. Current cron job ID `324942df`. Stops on session close or
via `CronDelete 324942df`.
