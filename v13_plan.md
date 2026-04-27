# v13 Plan — Strip-down + Capacity Recovery

**Date:** 2026-04-27
**Branch:** shikhar
**Predecessor:** v12 bankless (sliding 1.0955, gap 0.0047, size 16.54 MB — over cap)
**Goal:** Land a submittable run (≤16 MB) at sliding ≤1.090 BPB, ideally closer to 1.083.
**Competition SOTA:** PR #1493 at 1.0810 BPB.

---

## Experiment Results Summary

### v13a — Strip VE (baseline, no TTT)
**Config:** VE_ENABLED=0, TARGET_MB=15.2, TTT_ENABLED=0, everything else same as v12.

| Metric | v12 | v13a | Delta |
|--------|-----|------|-------|
| post_ema val_bpb | 1.0908 | **1.0901** | -0.0007 (better) |
| val_loss | 2.8177 | 2.8174 | -0.0003 |
| final_int6_roundtrip_exact | 1.1050 | **1.1050** | 0.0000 |
| sliding BPB | 1.0955 | **1.0955** | 0.0000 |
| size (bytes) | 16,537,005 | **15,844,725** | -692,280 |
| submittable? | NO (over cap) | **YES** | |
| training steps | ~4836 | 5108 | +272 (fewer params = faster steps) |

**Finding: VE strip is essentially free.** Val_loss identical, post-EMA actually 0.0007 better due to 272 extra training steps from faster step time. Size comfortably under 16MB cap with no pruning. This is our submittable baseline.

### v13b — Strip Bigram (VE kept) — SKIPPED
Skipped in favor of testing TTT directly on v13a config. The v13b run scripts exist but were not completed with the correct tokenizer.

### v13c — TTT on Bankless (first test)
**Config:** Same as v13a + TTT_ENABLED=1, TTT_LR=0.005, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768, TTT_MOMENTUM=0.9.

Training phase identical to v13a (TTT only affects eval):
- post_ema val_bpb: 1.0897 (matches v13a within noise)
- final_int6_roundtrip_exact: 1.1067

**TTT eval result:**

| Metric | Without TTT (INT6) | With TTT | Delta |
|--------|-------------------|----------|-------|
| BPB | 1.1067 | **1.7519** | **+0.6452 (catastrophic)** |
| Eval time | ~9s | **571s (~9.5min)** | 63x slower |

**Finding: TTT is DEAD for our stack.** This is the worst TTT failure across all versions:
- v4 (bank-based): +0.089 BPB
- v5 (bank-based): +0.046 BPB  
- v13c (bankless): **+0.645 BPB** — complete divergence

The SGD adaptation catastrophically diverges on the bankless architecture. TTT also nearly exceeds the 10-minute eval cap by itself (571s). **Do not attempt TTT again without fundamental redesign.**

### v13d — Parallel Residuals — NOT YET RUN
Run script exists at `run_v13d_parallel.sh`. PARALLEL_START_LAYER=7 (layers 7-10 run attn+MLP in parallel). Zero parameter cost, but unknown BPB impact on our stack.

---

## Key Learnings & Bug Fixes

### 1. Tokenizer Mismatch (CRITICAL)
Initial v13a run used wrong tokenizer file (370,917 bytes from `records/` folder) instead of the correct one (370,952 bytes from `kevclark/parameter-golf` HF repo at `datasets/tokenizers/fineweb_8192_bpe.model`).

- Wrong tokenizer: bytes_per_token = 3.527 → all BPB numbers ~0.06 higher than reality
- Correct tokenizer: bytes_per_token = 3.727 → matches v12's reported numbers

**Lesson:** Always trace the actual tokenizer path from the run script, don't copy from unrelated directories. The v12 script uses `data/tokenizers/fineweb_8192_bpe.model` — verify the file hash matches before running.

### 2. TARGET_MB Units Bug
Code interprets TARGET_MB in MiB (×1,048,576 bytes), but competition cap is 16,000,000 decimal bytes.
- 15.2 MiB = 15,938,355 bytes → 61,645 bytes of headroom under 16MB cap
- TARGET_MB must be ≤15.25 to be safe

### 3. TTT inference_mode Bug (Fixed in train_v13.py)
TTT eval crashed with `RuntimeError: Inference tensors cannot be saved for backward` due to two sources of inference-mode RoPE tensor caching:

**Bug 1:** Phase 1 (scoring) in `eval_val_sliding_ttt` used `torch.inference_mode()`, caching RoPE cos/sin as inference tensors. Phase 2 (training) couldn't backprop through them.
- **Fix:** Changed Phase 1 from `torch.inference_mode()` to `torch.no_grad()` (line 1239).

**Bug 2:** The INT6 roundtrip eval (line 1459) runs under `inference_mode()` before TTT eval is called, also poisoning the RoPE cache.
- **Fix:** Added RoPE cache invalidation (`m._seq_len_cached = 0` on all `Rotary` modules) before Phase 2 training loop.

Both fixes are in `train_v13.py`. These fixed the crash but TTT still diverged catastrophically on BPB.

### 4. VE Strip is Essentially Free
Pre-plan estimates suggested VE strip could cost 0.005-0.015 BPB. Actual cost: **0.000 BPB** on val_loss, with post-EMA actually slightly *better* due to recouped training steps. The VE embedding at layers 9,10 was not load-bearing for this architecture.

---

## Original Plan (for reference)

### Two coupled goals (and the tension between them)

1. **Bring pre-quant down from 1.0908 → ~1.083-1.085** (close 0.006-0.008 BPB)
2. **Get under the 16 MB submission cap** (drop ≥0.6 MB to leave wrapper headroom)

### What v12 carries (vs PR #1394 / #1493 base)

PR #1394 base: 11 layers, model_dim=512, SP8192 vocab, recurrence on 4-5, parallel residuals optional, MuonEq-R, SDClip k=12.85. **No** SWA, **no** bigram, **no** VE, **no** SmearGate, **no** PKO, **no** XSA, **no** TTT.

v12's extras on top of that base (each is a strip candidate):

| Feature | v12 setting | Estimated size | Actual strip cost (BPB) |
|---|---|---|---|
| **ValueEmbedding** (`ve_shared`) | enabled, vocab×128, layers 9,10 | ~1.0 MB | **0.000** (free!) |
| **BigramHashEmbedding** | vocab=3072, dim=112 | ~0.40 MB | Not tested |
| **SmearGate** | per-block gate | ~0.05 MB | Not tested |
| **SWA** (window=256, layers 0-5) | on | 0 MB | Not tested |
| **XSA_LAST_N=11** | on | 0 MB | Not tested |
| **PKO** | on | 0 MB | Not tested |
| **TTT** | disabled | 0 MB | **+0.645 catastrophic** |

### What we're missing vs PR #1493 (1.0810 SOTA)

| Feature | v12 has | PR #1493 | Status |
|---|---|---|---|
| **3-layer recurrence (L3-5)** | No (L4-5) | Yes | Not tested |
| **Parallel residuals from L7+** | Off | On | v13d script ready, not run |
| **TTT (SGD)** | Disabled | On | **DEAD — diverges on bankless** |
| **Hyperparams: WD=0.095, MLR=0.022** | Different | Tuned | Not tested |

---

## Current Status & Next Steps

**Best submittable result:** v13a at sliding 1.0955, size 15,844,725 bytes.
**Gap to SOTA:** 1.0955 - 1.0810 = 0.0145 BPB.

### Remaining experiments to try:
1. **v13d — Parallel residuals** (PARALLEL_START_LAYER=7): Free parameter cost, script ready
2. **3-layer recurrence** (RECUR_LAYERS="3,4,5"): More virtual depth, risk of reopening quant gap
3. **Hyperparameter tuning** (WD, MLR, EMA): Align with PR #1493's tuned values
4. **Strip bigram** (if we need more size headroom for adding capacity elsewhere)

### Dead ends (do not revisit):
- **TTT on bankless**: Catastrophic divergence (+0.645 BPB), 63x slower eval
- **TTT on bank-based**: Already failed in v4 (+0.089) and v5 (+0.046)

---

## Files

| File | Purpose |
|------|---------|
| `train_v13.py` | Training script (copied from train_v12_bankless.py + TTT bug fixes) |
| `run_v13a_strip_ve.sh` | Strip VE config — **submittable baseline** |
| `run_v13b_strip_bigram.sh` | Strip bigram config (not run with correct tokenizer) |
| `run_v13c_ttt.sh` | TTT test config (DEAD) |
| `run_v13d_parallel.sh` | Parallel residuals config (not yet run) |
| `v13a_strip_ve.log` | v13a run 1 (wrong tokenizer) |
| `v13a_strip_ve_r2.log` | v13a run 2 (correct tokenizer) — **reference run** |
| `v13c_ttt_run3.log` | v13c TTT final run (catastrophic result) |
