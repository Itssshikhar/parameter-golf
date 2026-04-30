# PR1493 Tuning Sweep — Session, 2026-04-30

Single-seed (`SEED=42`) hparam sweep on top of `wd_paired` (entry #6, q_ttt 1.07974).
Goal: find an env-var-only win that clears `q_ttt ≤ 1.07935` so we can stop and
move to seed proof + code shrink. If nothing clears, fall back to optimizer-geometry
code work (paired-head qkv mode, Polar Express, NorMuon).

## Setup

- 8× H100 80GB HBM3, torch 2.9.1+cu128, flash_attn_3 cu128_torch291.
- `train_pr1493.py` taken from `origin/shikhar` HEAD (`468be92`, file md5 `aa4d62de…`).
  Note: NOT byte-identical to the 2026-04-29 `74dc702` build (md5 `968e5ab7…`)
  used for the `wd_paired` 1.07974 record. The 468be92 version adds GPTQ all-reduce
  / damp / block_size knobs over 74dc702. Smoking-gun string `tagged=22` confirmed
  on every run.
- Dataset: `kevclark/parameter-golf` SP8192, 128 train shards, 1 val shard,
  fresh download.
- Brotli + zstandard installed locally; `safe_launch.sh` symbol check passes
  on every launch.
- Common env: `SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5
  WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1`.
- No wd_paired anchor was rerun on this box (skipped by user request to save GPU).
  All deltas in this doc are vs the recorded 2026-04-29 wd_paired numbers
  (entry #6: pre 1.08610, q_sw 1.08209, q_ttt 1.07974). With a different file md5
  and fresh data download, machine-side drift cannot be ruled out at the 0.0002 level.

## Plan

| # | Run | Override on top of wd_paired | Prior strength |
|---|-----|------------------------------|----------------|
| R1 | gptq_buy | `GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4` | mechanistic — buys ~+8 s = ~+60 steps |
| R2 | ema9970 | `EMA_DECAY=0.9970` | hparam, no theory |
| R3 | minlr05 | `MIN_LR=0.05` | modded-nanogpt precedent |
| R4 | loop040 | `ENABLE_LOOPING_AT=0.40` | unprobed under wd_paired |
| R5 | qk500 | `QK_GAIN_INIT=5.00` (vs 5.25) | small perturbation |

Skipped from the original list: `PAIRED_HEAD_MUON_MODE=qkv` — that env var doesn't
exist; tagging V by KV-heads is real code work (V is at KV-head granularity, O is at
full-head granularity, naive grouping silently corrupts grads). Deferred to the
fallback branch if the env-var sweep produces no winner.

Single seed per cell — the user explicitly accepted that low-signal cells (R5,
R4) won't be distinguishable from noise; they're run as cheap negative filters.
3-seed promotion gated on `q_ttt ≤ 1.07935`.

## Results

### R1 — gptq_buy (done)

```
Override: GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4
```

| metric | R1 | recorded wd_paired | Δ |
|--------|----|--------------------|---|
| stop_step | 4653/20000 | 4596/20000 | +57 |
| pre | 1.08579 | 1.08610 | **−0.00031** |
| q | 1.09864 | 1.09891 | −0.00028 |
| q_sw | 1.08184 | 1.08209 | −0.00025 |
| **q_ttt** | **1.07952** | **1.07974** | **−0.00022** |
| size | 16,031,834 B | 16,029,924 B | +1,910 B |

Sanity: 67 Hessians in **3.5 s** (vs 12.8 s for batches=64) — 4× faster as
expected. Effective training cap ran 596 000 ms vs baseline 588 000 ms (+8 s as
predicted). The +57 extra training steps converted into a uniform improvement
across pre/q/q_sw/q_ttt.

**Verdict.** Real win, transfers cleanly through quantization. Passes pre gate
(−0.00031 ≥ 0.00015) and q_sw-not-worse gate. **Fails the serious-candidate gate**
(q_ttt 1.07952 > 1.07935 by +0.00017). On its own, +0.00022 BPB is at the
single-seed noise floor for this stack — needs to stack with another candidate
or get re-confirmed across seeds before treating as real.

### R2 — ema9970 (running)

### R3 — minlr05 (pending)

### R4 — loop040 (pending)

### R5 — qk500 (pending)

## Fallback plan if env-var sweep produces no clear winner

User-confirmed sequence after R1–R5 finish:

1. Implement `PAIRED_HEAD_MUON_MODE=qkv` — group `c_v.weight` by KV heads in the
   paired-head NS tag. `c_o` deferred (mixes heads intentionally, qkvo is risky).
2. Polar Express NS coefficients in `Muon.zeropower_via_newtonschulz5` (replace
   the current `(3.4445, -4.7750, 2.0315)` triplet).
3. NorMuon-style normalization (per-iter normalization in the NS loop).

These are optimizer-geometry changes, not hparam knobs. Same class of idea that
produced paired-head Muon in the first place. Each will need its own session doc.
