# PR1493 wd_paired_iha — Session, 2026-04-29

Quick gate-test of stacking IHA on top of `wd_paired`. Killed at the pre-quantization post-EMA gate per agreed criterion.

## Setup

```bash
RUN_ID=pr1493_wd_paired_iha_s42 SEED=42 QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
IHA_ENABLED=1 \
./safe_launch.sh torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

`IHA_MIX_V` left at default (0): IHA on Q/K only, V passthrough.

`safe_launch.sh` reported `HEAD=2a8fbca file_md5=968e5ab…` at launch.

## Live evidence

```text
iha_enabled: True
iha_mix_v: False
paired_head_muon_enabled: True
wd_schedule_enabled: True
wd_sched_low_factor: 0.65   # default (not strong)
wd_sched_high_factor: 1.5   # default
train_shards: 128
muon:paired-head NS enabled for q/k matrices tagged=22
```

All flags confirmed live in the hyperparameter dump and on every rank's `/proc/<pid>/environ`. Training proceeded normally.

## Train cap and pre-quant

```text
stopping_early: wallclock_cap train_time: 588104ms step: 4528/20000
ema:applying EMA weights
pre-quantization post-ema val_loss:2.80695725 val_bpb:1.08666062 eval_time:7889ms
iha:folded active head mixes into linear weights for 11 layers before GPTQ
```

**`pre = 1.08666`**

Comparison to gate threshold:
- Kill if `pre ≥ 1.08610` (wd_paired's pre)
- Continue if `pre ≤ 1.08580` (wd_strong_paired's pre, with small buffer)
- 1.08666 is **0.00056 above** the kill threshold.

Comparison to all peers:
- baseline: 1.08757
- wd alone: 1.08650
- **wd_paired: 1.08610**
- wd_strong_paired: 1.08573
- **wd_paired_iha: 1.08666** ← worse than every alternative including wd alone

Killed via SIGTERM to torchrun parent before GPTQ completed. Confirmed all 8 worker PIDs cleared and GPU memory freed.

## Why IHA hurt

Two factors compose:
1. **IHA's per-step forward overhead** — the inline `F.linear(x, mixed_weight)` adds compute per layer-block. Wallclock cap hit at step **4528** (vs wd_paired's 4596 = **−68 steps**, ≈1.5% less training).
2. **IHA changes the trained representation** in a way that doesn't help once you've already added paired-head Muon. q_mix/k_mix were learned as identity-ish anyway (we only mix within heads), so the optimization landscape change is small but the per-step cost is not.

The `iha:folded ... before GPTQ` line confirms the **harness fix** (the prior session's `KeyError: 'blocks.0.attn.c_q.weight'`) is correct: `fold_iha_mixes` correctly folded the head mixes into the underlying Q/K linear weights before GPTQ Hessian collection. So IHA itself is **runnable** end-to-end now; it just isn't a stacking win on top of `wd_paired`.

A small note: `passthrough (float16): blocks.attn.k_mix, blocks.attn.q_gain, blocks.attn.q_mix, ...` — q_mix/k_mix params remain in the state dict as float16 even after the fold. Adds ≈few KB to submission size on top of the regular cost (post-fold the mix matrices are no-ops at inference time, but they're still serialized).

## Verdict

IHA harness is fixed but does **not** stack with `wd_paired`. Drop. Move to step 2 in the plan: code-shrink scout, then 3-seed `wd_paired` sweep on the trimmed script.

## Files committed

- `pr1493_wd_paired_iha_session.md` — this document.
- `pr1493_priority_results.md` — appended row 8.
- `logs/pr1493_wd_paired_iha_s42.txt` — partial log (train + pre-quant + GPTQ Hessian collection through kill).
- `logs/pr1493_wd_paired_iha_s42.stdout` — torchrun stdout including the safe_launch line and the SIGTERM traceback.
