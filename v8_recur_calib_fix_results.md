# v8: Recurrence Hessian Calibration Fix — Results

**Date:** 2026-04-27
**Branch:** shikhar
**RUN_ID:** `v8_recur_calib_fix`
**Hardware:** 8xH100 SXM 80GB (local container)

## Hypothesis

The GPTQ calibration model (`_HessianGPT`) was not applying depth recurrence during Hessian collection, causing a mismatch between calibration-time and eval-time activation distributions. Fixing this mismatch should substantially reduce the quantization gap observed in v7 runs (0.08-0.14 BPB).

## What was changed (on top of the recurrence_hessian_fix.md edits)

1. **Partial key offset in `_HessianAttn`** — Added `self.partial_key_offset` attribute and the K-shift logic (`k[:, 1:, :, rd:] = k[:, :-1, :, rd:]`) to `_HessianAttn.forward`, matching `CausalSelfAttention.forward`. PKO is enabled on full-attention layers (6-10) in `_HessianGPT.__init__`, same gating logic as `GPT.__init__`. Without this, Hessians for layers 6-10 would be collected without the key shift that training and eval use — the same class of calibration mismatch the recurrence fix addresses.

2. **AR generation recurrence log line** — Added `log0(f"gptq:AR generation base_model.recur_active={base_model.recur_active}")` before `generate_autoregressive_calib` to confirm that AR calibration tokens are generated with recurrence active.

## Config

```bash
RUN_ID=v8_recur_calib_fix SEED=1337 \
TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 \
SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5 \
PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
RECUR_LAYERS="4,5" RECUR_START_FRAC=0.5 \
WARMDOWN_ITERS=4000
```

## Results

| Metric | Value |
|--------|-------|
| Steps completed | 5194/20000 (wallclock capped) |
| step_avg | 115.54ms |
| Recurrence activated | Step 2926 @ 300s |
| SWA start | Step 4450 |
| Late QAT enabled | Step 4607 (scale 0.1499) |
| Mid-train val_bpb (step 4000) | 1.1286 |
| Final val_bpb (step 5194) | 1.0872 |
| Post-EMA val_bpb | **1.0862** |
| INT6 roundtrip val_bpb | **1.1887** |
| Sliding window eval val_bpb | **1.1795** |
| Quant gap (roundtrip) | 0.1025 |
| Quant gap (sliding) | 0.0933 |

### Log confirmations

- `gptq:enabled recurrence on hessian_model (recur_layers=4,5)` — Hessians collected with recurrence ✓
- `gptq:AR generation base_model.recur_active=True` — AR tokens generated with recurrence ✓
- `gptq:collected hessians for 68 layers (AR self-gen)` — same layer count as previous runs ✓

## Comparison to all recurrence runs

| Run | Fix applied | Pre-quant | Roundtrip | Sliding | Quant gap (sliding) |
|-----|------------|-----------|-----------|---------|-------------------|
| v3 (no fix, seq2048) | None | 1.0949 | — | 1.1792 | 0.0843 |
| v7a (INT8 recurred) | INT8 layers 4,5 | 1.0888 | 1.2263 | — | 0.1375 |
| v7b (skip recur eval) | No recur at eval | 1.0874 | 1.1961 | — | 0.1087 |
| v7c (INT8 + skip eval) | Both | 1.0886 | 1.2251 | — | 0.1365 |
| **v8 (this run)** | **Hessian recur + PKO** | **1.0862** | **1.1887** | **1.1795** | **0.0933** |

### Comparison to non-recurrence baselines

| Run | Pre-quant | Sliding | Quant gap |
|-----|-----------|---------|-----------|
| v6 (no recurrence, seq2048) | 1.0986 | 1.1029 | 0.0043 |
| Bank QAT (no recurrence, seq4096) | 1.1200 | 1.1112 | 0.0051 |
| **v8 (recurrence, our stack)** | **1.0862** | **1.1795** | **0.0933** |

## Analysis

### The calibration fix did not work

The quant gap is 0.0933 — essentially identical to v3's 0.0843 and still catastrophic (20x the non-recurrence baseline of 0.004-0.005). The Hessian mismatch was a real bug but fixing it had negligible impact.

---

## Competition PR Analysis: How SOTA Handles Recurrence + Quantization

### PR #363 — The Foundational Discovery (evangelinehelsinki, Mar 25 2026)

**"Non-record: Depth Recurrence in Parameter-Constrained Transformers — What Works, What Doesn't, and Why"**

This non-record submission ran ~35 experiments and documented the core problem:

> **"Quantization error amplifies approximately 900× through recurrence cycles. Shared weights quantized once propagate error superlinearly across repetitions, making int6 incompatible with weight-sharing architectures."**

Key findings:
- Stem-core-tail architecture: 1-3 unique → 2-3 shared blocks repeated 2-5× → 1-3 unique
- 3×3 > 2×5 loops (more unique blocks with fewer repeats wins)
- Int8 for shared blocks mitigates the amplification
- **Noisy QAT (differentiable uniform noise during training) reduced quant gap from 0.37 BPB to 0.002 BPB** — the breakthrough technique

This is the key finding we missed: the fix for recurrence + quantization is NOT better calibration — it's **aggressive QAT that makes weights inherently robust to quantization noise**.

### PR #1394 — SDClip + Loop45x2 (clarkkev, Apr 5 2026, 1.0856 BPB)

**"SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip"**

Architecture:
- 11 physical layers, layers 4-5 looped once (13 virtual layers)
- SP8192 vocabulary
- MuonEq-R optimizer
- No SWA, no bigram, no value embeddings, no PKO

**Critical implementation detail: Hessians are collected WITHOUT recurrence (`looping_active=False`).**

```python
# Their forward pass:
enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
dec_iter = self.decoder_indices if self.looping_active else range(...)

# During Hessian collection, model.looping_active is False.
# Hessians see each layer ONCE, not twice.
```

Quantization:
- SDClip: clip threshold = k × std(row), where k=12.85 for matrices, k=20.0 for embeddings
- Full Hessian GPTQ with Cholesky decomposition
- Single model class — no separate `_HessianGPT`
- Training data for calibration (not AR self-gen)
- Quant gap: ~0.012 BPB

**This proves the Hessian recurrence mismatch is irrelevant.** They deliberately skip recurrence during calibration and still get a small gap.

### PR #1204 — ParallelResiduals + MiniDepthRecurrence (msisovic, Apr 9 2026, 1.1063 BPB)

- Layers 4,5 recur (same as ours)
- Recurrence activates at step 3000
- Uses AR self-gen GPTQ calibration (like us)
- Mixed quantization: 32 int6 layers + remaining int8
- Achieved 1.1063 BPB

### PR #1285 — MuonEq-R + Depth Recurrence (dexhunter, Apr 9 2026, 1.0912 BPB)

- Layers 4,5 recur
- **All 66 layers at int6** (no mixed int5/int8)
- Higher weight decay (0.090) produces smaller magnitudes → better compression → room to keep everything int6
- Brotli compression
- 1.0912 BPB

### PR #1334 — SP4096 + Depth Recurrence (aryanbhosale, Apr 9 2026, 1.0897 BPB)

- RECUR_LAYERS=4,5, RECUR_START_STEP=3000
- PARALLEL_START_LAYER=7
- Uses PR #1394's codebase (same quantization pipeline)
- Same approach: Hessians collected WITHOUT recurrence
- 1.0897 BPB

### PR #1493 — Current SOTA (bigbag, Apr 9 2026, 1.0810 BPB)

- **3-layer recurrence on L3-5** (17 virtual layers from 11 physical)
- Activates at 0.35 wallclock fraction
- Parallel residuals from L7+
- QK-Gain 5.25
- Legal score-first TTT (SGD lr=0.005, 3 epochs)
- Tuned hyperparameters: WD=0.095, MLR=0.022, EMA=0.9965
- 1.0810 BPB (3-seed mean, std 0.0002)

---

## Why Competition Stacks Work and Ours Doesn't

### Finding 1: Hessian mismatch is irrelevant

PR #1394, #1334, and #1493 all collect Hessians **without recurrence active** — the exact "bug" we tried to fix. They get quant gaps of 0.01-0.02. Our v8 fixes the mismatch and still gets 0.09. **The calibration path is not the problem.**

### Finding 2: The real fix is QAT, not calibration

PR #363 documented the root cause: quantization error amplifies ~900× through recurrence. Their solution — Noisy QAT with differentiable uniform noise — collapsed the gap from 0.37 to 0.002 BPB. The competition stacks that followed (#1394, #1285, etc.) evolved this into their training pipelines.

Our late QAT runs for only ~560 steps (out of 5194 total) at the very end of training. The competition stacks likely have QAT active for much longer, or use a different QAT approach that better conditions the weights for quantization robustness.

### Finding 3: No parameter banks in competition stacks

Every competition PR uses a **single model class** with standard per-layer `nn.Linear` / `CastedLinear` weights. No 3D parameter banks, no unbank/rebank pipeline, no separate `_HessianGPT`.

Our stack's parameter bank architecture means:
1. Weights must be unbanked (3D → 2D) before quantization
2. Quantized weights must be rebaned (2D → 3D) for eval
3. The `_HessianGPT` exists because the real model can't run hooks on standard Linear layers
4. Any subtle error in unbank/rebank is invisible without recurrence (each weight used once) but amplified 900× with recurrence (weights reused)

### Finding 4: Simpler architectures survive recurrence better

Competition stacks have: no SWA, no bigram, no value embeddings, no PKO, no smear gate. Each additional feature adds weight complexity that is fine at 0.004 quant gap but becomes catastrophic at 0.09 when error amplifies through recurrence.

### Finding 5: SDClip may be more recurrence-friendly than our quantization

PR #1394's SDClip (`k × std(row)`) is a principled per-row clip threshold. Our quantization uses percentile-based search or Hessian-weighted GPTQ. SDClip's monotonic relationship between k and clip range may produce more stable quantization for weights that serve dual activation distributions.

---

## Root Cause Assessment (revised)

The Hessian calibration mismatch was **not** the cause of our 0.09 BPB quant gap. The actual causes, in order of likely importance:

1. **Insufficient QAT** — Our late QAT (~560 steps) is far too short to condition weights for quantization robustness under recurrence. PR #363 showed that proper QAT collapses the gap from 0.37 to 0.002.

2. **Parameter bank architecture** — The unbank → quantize → rebank pipeline may introduce errors that amplify through recurrence. Competition stacks avoid this entirely by using standard per-layer weights.

3. **Architectural complexity** — SWA, bigram, VE, PKO, and smear gate add weight structure that may be more sensitive to quantization noise under recurrence.

4. **Quantization method** — SDClip may be inherently more stable than our GPTQ approach for recurred layers.

## Recommended Next Steps

1. **Port competition QAT**: Implement PR #363's Noisy QAT (differentiable uniform noise, not just STE fake quantize). Enable from recurrence activation onward (~step 2926), not just the last 560 steps.

2. **Try SDClip**: Replace our percentile/Hessian-weighted GPTQ with `k × std(row)` clipping for at least the recurred layers.

3. **Strip to minimal recurrence stack**: Test recurrence on a simplified version of our code — no SWA, no bigram, no VE, no PKO — to isolate whether the architecture or the quantization pipeline is the bottleneck.

4. **Investigate unbank/rebank fidelity**: Add assertions that `unbank(bank(x)) == x` for the recurred layer weights specifically, to rule out numerical drift in the bank pipeline.

## Artifacts

- Log: `v8_recur_calib_fix.log`
- Code changes: `train_gpt_swa.py` (PKO in `_HessianAttn`, recurrence log line)
- Fix doc: `recurrence_hessian_fix.md`

---

## Experiment A1: Longer STE QAT (with recurrence timing)

**Date:** 2026-04-27
**Script:** `train_recur_qat.py`
**Config:** Same as v8 + `QAT_WITH_RECURRENCE=1 NOISY_QAT=0`

QAT activated at step 2931 (with recurrence) — ~2242 STE QAT steps vs v8's ~560.

| Metric | A1 (longer STE) | v8 (baseline) | Delta |
|--------|-----------------|---------------|-------|
| Post-EMA pre-quant | 1.0945 | 1.0862 | +0.0083 (worse) |
| INT6 roundtrip | 1.1854 | 1.1887 | -0.0033 (tiny improvement) |
| Quant gap (roundtrip) | 0.0909 | 0.1025 | -0.0116 |

**Verdict: Failed.** Longer STE QAT marginally reduced the quant gap (0.0909 vs 0.1025) but cost 0.008 in pre-quant quality. Net roundtrip barely moved (-0.003). STE fake quantization cannot train weights to be robust to the 900× error amplification through recurrence — the gradients through the STE detach are too coarse.

## Experiment A2: Noisy QAT (differentiable uniform noise, with recurrence timing)

**Date:** 2026-04-27
**Script:** `train_recur_qat.py`
**Config:** Same as v8 + `QAT_WITH_RECURRENCE=1 NOISY_QAT=1`

Noisy QAT activated at step 2923 (with recurrence) — ~2023 steps of differentiable uniform noise QAT.

| Metric | A2 (noisy QAT) | A1 (longer STE) | v8 (baseline) |
|--------|-----------------|-----------------|---------------|
| Post-EMA pre-quant | 1.0942 | 1.0945 | 1.0862 |
| INT6 roundtrip | 1.1745 | 1.1854 | 1.1887 |
| Sliding eval | **1.1665** | 1.1768 | 1.1795 |
| Quant gap (sliding) | **0.0723** | 0.0823 | 0.0933 |

**Verdict: Marginal improvement, not a fix.** Noisy QAT reduced the quant gap from 0.0933 → 0.0723 (22% relative improvement). Sliding eval improved by 0.013 BPB. But the gap is still 0.07 — 18x the non-recurrence baseline (0.004). The absolute sliding eval of 1.1665 is far worse than our non-recurrence best of 1.1112.

## Full Experiment Summary

| Run | QAT method | QAT steps | Pre-quant | Sliding | Quant gap |
|-----|-----------|-----------|-----------|---------|-----------|
| v8 | STE | ~560 | 1.0862 | 1.1795 | 0.0933 |
| A1 | STE | ~2242 | 1.0945 | 1.1768 | 0.0823 |
| A2 | Noisy | ~2023 | 1.0942 | 1.1665 | 0.0723 |
| Non-recurrence best | Bank QAT | ~560 | 1.1200 | 1.1112 | 0.0051 |
| PR #1394 (competition) | Unknown | Unknown | — | 1.0856 | ~0.012 |

## Conclusion: The Parameter Bank Architecture is the Primary Suspect

QAT improvements (both duration and method) provide diminishing returns:
- 4x more STE QAT steps: gap 0.0933 → 0.0823 (-12%)
- Noisy QAT: gap 0.0933 → 0.0723 (-22%)

Neither comes close to the competition's 0.01-0.02 gap. Since the competition stacks use standard per-layer weights (no parameter banks), the unbank → quantize → rebank pipeline is the most likely remaining differentiator. Next step: investigate whether the bank pipeline introduces numerical errors that compound through recurrence.

---

## Experiment v9: Competition Quantization Pipeline

**Date:** 2026-04-27
**Script:** `train_gpt_swa.py` (modified)
**Config:** Same as v8 + competition's quantization pipeline

### What was changed (on top of v8):

1. **SDClip k=12.85** (was 14.0) — matches PR #1394/#1493 exactly. Line 572: `MATRIX_CLIP_SIGMAS` default changed from 14.0 to 12.85.

2. **Training data for calibration** (was AR self-gen) — matches competition. New env var `CALIB_MODE` defaults to "train" (uses `collect_hessians` with `train_loader`, 256 batches). Old AR self-gen available via `CALIB_MODE=ar`.

3. **Recurrence ON during calibration** — matches competition. New env var `CALIB_RECURRENCE` defaults to "1". The agents that fetched PR #1394 and #1493's actual code confirmed that competition KEEPS `looping_active=True` during Hessian collection — contradicting our earlier analysis that said they collected without recurrence.

4. **Embedding GPTQ** — matches PR #1394's "GPTQ Embeddings" approach. Added output hook on `hessian_model.final_norm` to collect H=X^T X from pre-logit activations. The embedding Hessian is used for GPTQ int8 quantization (clip_range=127, k=20.0) instead of simple round-to-nearest. Competition applies full GPTQ error compensation to the tied embedding.

5. **No pruning needed** — With k=12.85 (tighter clip), the quantized model fits at 15.84MB (under 15.9MB target). No magnitude pruning required, matching competition which also fits without pruning.

### Critical correction from actual PR code:

Our earlier analysis (in the "Competition PR Analysis" section above) stated that PR #1394 collected Hessians WITHOUT recurrence. This was WRONG. Agents that fetched and analyzed the actual `train_gpt_human.py` from both PRs confirmed:

- PR #1394: `looping_active` stays True from training, never explicitly disabled before `collect_hessians`. Hessians collected WITH recurrence.
- PR #1493: Same — `looping_active` stays True. Hessians collected WITH recurrence.

The earlier claim was based on seeing `enc_iter = self.encoder_indices if self.looping_active else range(...)` and incorrectly assuming `looping_active` was set to False before calibration.

### Config:
```bash
RUN_ID=v9_competition_quant SEED=1337 \
TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 \
SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5 \
PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
RECUR_LAYERS="4,5" RECUR_START_FRAC=0.5 \
WARMDOWN_ITERS=4000 MATRIX_CLIP_SIGMAS=12.85 \
torchrun --standalone --nproc_per_node=8 train_gpt_swa.py
```

### Results:

| Metric | Value |
|--------|-------|
| Steps completed | 5396/20000 (wallclock capped) |
| step_avg | 111.20ms |
| Recurrence activated | Step 2925 @ 300s |
| SWA start | Step 4650 |
| Late QAT enabled | Step 4833 (scale 0.1498) |
| Mid-train val_bpb (step 4000) | 1.1352 |
| Final val_bpb (step 5396) | 1.0861 |
| Post-EMA val_bpb | **1.0849** |
| INT6 roundtrip val_bpb | **1.1758** |
| Sliding window eval val_bpb | **1.1668** |
| Quant gap (roundtrip) | 0.0909 |
| Quant gap (sliding) | 0.0819 |

### Log confirmations:
- `gptq:enabled recurrence on hessian_model (recur_layers=4,5)` — Recurrence ON ✓
- `gptq:registered embedding Hessian hook on final_norm (competition GPTQ-Embeddings approach)` — Embedding GPTQ ✓
- `gptq:collecting hessians from training data (competition approach, 256 batches)` — Training data calibration ✓
- `gptq:collected hessians for 68 layers (training data) in 61.3s` — 68 layers ✓
- `gptq:collected embedding Hessian from 256 batches` — Embedding Hessian collected ✓
- `prune: already fits, no pruning needed` — No pruning ✓

### Updated Full Experiment Comparison:

| Run | Pipeline changes | Pre-quant | Sliding | Quant gap (sliding) |
|-----|-----------------|-----------|---------|-------------------|
| v3 (no fix, seq2048) | None | 1.0949 | 1.1792 | 0.0843 |
| v7a (INT8 recurred) | INT8 layers 4,5 | 1.0888 | — | — |
| v7b (skip recur eval) | No recur at eval | 1.0874 | — | — |
| v8 (Hessian recur + PKO) | Hessian recur fix | 1.0862 | 1.1795 | 0.0933 |
| A1 (longer STE QAT) | 4x STE QAT steps | 1.0945 | 1.1768 | 0.0823 |
| A2 (noisy QAT) | Differentiable noise QAT | 1.0942 | 1.1665 | 0.0723 |
| **v9 (competition quant)** | **Train data, k=12.85, emb GPTQ, no prune** | **1.0849** | **1.1668** | **0.0819** |
| Non-recurrence best | Bank QAT | 1.1200 | 1.1112 | 0.0051 |
| PR #1394 (competition) | Their full stack | 1.0899 | 1.0866 | ~0.003 |
| PR #1493 (SOTA) | Their full stack | 1.0873 | 1.0829 | ~0.004 |

### Analysis:

**Matching competition's entire quantization pipeline did not close the gap.** The quant gap moved from 0.0933 (v8) to 0.0819 (v9) — a 12% relative improvement, but still 20x competition's ~0.004.

We have now eliminated EVERY post-training variable:
1. ✅ Calibration data source → training data (matches competition)
2. ✅ Calibration recurrence → ON (matches competition)
3. ✅ SDClip k → 12.85 (matches competition)
4. ✅ Embedding GPTQ → added (matches competition)
5. ✅ Pruning → none needed (matches competition)

The remaining differences are all in the **model architecture and training**:
- Parameter banks (3D weight tensors) vs competition's per-layer nn.Linear
- Paired-head Muon vs competition's MuonEq-R
- SWA on recurred layers (window=256) vs competition's full attention
- Bigram, VE, PKO, XSA, SmearGate — features competition doesn't use
- Weight decay 0.085 vs competition's 0.090-0.095

### Root Cause Assessment (final):

The quantization pipeline is definitively NOT the cause. The root cause is in how the weights are trained — the architecture and optimizer produce weight distributions that don't survive INT6 quantization under recurrence.

### Recommended Next Step:

Strip the two highest-suspicion architectural factors and retrain:
1. **SWA off** (SWA_WINDOW_SIZE=0) — recurred layers 4,5 switch to full attention, matching competition
2. **Weight decay 0.095** (MUON_WD=0.095) — matches competition, produces smaller/more quantization-friendly weights

If the gap closes → SWA/WD was the culprit, reintroduce features one by one
If the gap stays → parameter banks are the bottleneck, need bankless model

### Artifacts:
- Log: `v9_competition_quant.log`
- Model: uploaded to HuggingFace as `v9_competition_quant.pt` and `v9_competition_quant.int6.ptz`
