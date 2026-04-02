# PR #1263: LeakyReLU2 + XSA-all + Full GPTQ + SLOT

**BPB: 0.9354** (3-seed mean, std 0.0032)
**Status:** OPEN
**Author:** xexyz (Andrew Hernandez)
**Date:** 2026-04-02

## Key Techniques

1. **SLOT (Softmax Logit Optimization at Test-time)** - The main novelty
   - Based on arXiv:2505.12392v2
   - Extracts frozen hidden states from last layer under `torch.no_grad()`
   - Optimizes per-sample additive delta `[bsz, 1, 512]` + per-sample logit bias `[bsz, 1, 1024]`
   - 16 AdamW steps with cosine LR schedule (0.008 -> 0.0008)
   - Scored-position mask: only last `stride` tokens per non-first window contribute to SLOT loss
   - Model weights completely frozen during SLOT
   - ~311s eval time per run

2. **QK-Gain init = 4.0** - Sharpened attention maps via learned per-head gain
3. **XSA on all 11 layers** (not just last N)
4. **Full GPTQ** - Hessian-based int6 quantization with Cholesky error compensation (32 calibration batches)
5. **Late QAT** at threshold 0.15

## Architecture
- 11L, dim=512, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)^2 MLP with 3x expansion
- SmearGate + BigramHash embedding augmentation
- U-Net skip connections
- ~27M parameters
- Muon + Adam optimizers, EMA (0.997) + Tight SWA

## What's Novel
- **SLOT implementation** is the key differentiator - drops from 1.1263 (sliding) to 0.9354 (SLOT)
- That's a -0.191 BPB improvement from eval-time optimization alone
- Per-sample delta + logit bias approach inspired by PR #1229
- QK-Gain 4.0 validated by PR #1125, PR #1176

## What's Borrowed
- Base architecture from PR #1019 (@abaybektursun)
- LeakyReLU^2 from PR #549
- XSA from PR #198
- GPTQ from merged SOTA stack

## Compatibility with SOTA Stack
- Highly compatible - built directly on the merged SOTA stack
- SLOT is purely an eval-time technique, orthogonal to training improvements
- Could be added to any existing model without retraining

## CRITICAL NOTE
- The 0.9354 BPB is extremely impressive but hinges on SLOT legality
- SLOT optimization runs ~311s per eval (within 10-min eval budget)
- No n-gram cache, no two-pass rescoring, no eval-time training data access
- PR #1240 questions whether SLOT violates causal dependence
