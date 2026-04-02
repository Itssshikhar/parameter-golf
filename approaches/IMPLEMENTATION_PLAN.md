# Implementation Plan -- 5 Training/Architecture Approaches

## Philosophy

All 5 approaches change how the model is **trained** or **structured**. None are eval-time tricks. Each modifies the training pipeline, optimizer, loss function, or model architecture to produce a fundamentally better model within the 16MB / 10-minute constraints.

---

## The 5 Approaches

| # | Name | Type | Expected BPB Gain | Risk | Lines Changed |
|---|------|------|------------------|------|---------------|
| 1 | Low-Rank Factored MLP | Architecture | -0.004 to -0.010 | MEDIUM | ~60-80 |
| 2 | Reptile Meta-Learning Warmdown | Training (schedule) | -0.003 to -0.008 | MEDIUM | ~75-90 |
| 3 | SVD + Quantized Factors | Architecture + Compression | -0.002 to -0.006 | MEDIUM | ~100-120 |
| 4 | Multi-Token Prediction + BPB Loss | Training (loss function) | -0.003 to -0.008 | MEDIUM | ~65-80 |
| 5 | Gram-Newton-Schulz + FP8 Training | Training (throughput) | -0.003 to -0.006 | LOW-MEDIUM | ~60-80 |

**Theoretical maximum (all 5, fully additive): -0.015 to -0.038 BPB -> target 1.077-1.100**
**Realistic (partial additivity, some interactions): -0.008 to -0.020 BPB -> target 1.095-1.107**

---

## Implementation Order

The order is determined by three factors:
1. **Risk**: lowest-risk approaches first (quick wins, validate infrastructure)
2. **Independence**: approaches that don't require retraining first
3. **Composability**: approaches that compose cleanly with each other

```
Phase 1: THROUGHPUT (fastest to implement, lowest risk, benefits all subsequent work)
  └── Approach #5: Gram-Newton-Schulz + FP8 Training     [4-6 hours]

Phase 2: LOSS FUNCTION (training change, but same architecture)
  └── Approach #4: Multi-Token Prediction + BPB Loss      [4-6 hours]

Phase 3: TRAINING SCHEDULE (training change, requires careful integration)
  └── Approach #2: Meta-Learning Warmdown                  [6-8 hours]

Phase 4: ARCHITECTURE (most invasive changes)
  ├── Approach #1: Low-Rank Deep Model                     [6-8 hours]
  └── Approach #3: SVD + Quantized Factors                 [6-8 hours]
```

**Rationale:**
- Phase 1 first because GNS+FP8 makes ALL subsequent training runs faster. Every experiment benefits.
- Phase 2 second because MTP+BPB loss is a pure training modification on the existing architecture. Easy to A/B test.
- Phase 3 third because meta-warmdown requires the training loop to be working correctly (build on Phase 1+2).
- Phase 4 last because architecture changes are the most invasive and interact with everything else.
- Approaches #1 and #3 are alternatives (both change architecture for more depth) -- implement both, pick the winner.

---

## Phase 1: Training Throughput (Approach #5)

### Step 1.1: Implement Gram-Newton-Schulz

**Files:** `train_gpt.py` lines ~96-109 (zeropower_via_newtonschulz5)

**Tasks:**
1. Replace `zeropower_via_newtonschulz5` with `zeropower_via_gns`:
   - Fuse the polynomial: compute `poly(A) = a*I + b*A + c*A@A` as a single matrix
   - Apply via one matmul: `X = poly(A) @ X` instead of `X = a*X + B@X`
   - Same interface, same output (to bf16 precision)

2. Correctness test:
   ```python
   G = torch.randn(512, 1536, device='cuda', dtype=torch.bfloat16)
   out_old = zeropower_via_newtonschulz5(G, steps=5)
   out_new = zeropower_via_gns(G, steps=5)
   assert (out_old - out_new).abs().max() < 1e-3
   ```

3. Speed benchmark:
   ```python
   import time
   for _ in range(100):
       zeropower_via_newtonschulz5(G, steps=5)
   torch.cuda.synchronize()
   # Compare with GNS
   ```

**Expected: ~5ms/step savings -> ~6% speedup**

### Step 1.2: Implement FP8 Training Matmuls

**Files:** `train_gpt.py` CastedLinear class (~line 509)

**Tasks:**
1. Modify `CastedLinear.forward()` to use `torch._scaled_mm` with FP8 casts
2. Add per-tensor dynamic scaling for activations and weights
3. Keep FP8 disabled for:
   - `tok_emb` (embedding layer, precision-sensitive)
   - `lm_head` (output projection, precision-sensitive)
   - Any layer during late QAT phase (QAT needs bf16 precision)
4. Add `FP8_TRAINING` env var toggle

**Critical dependency:** Verify `torch._scaled_mm` is available in the RunPod PyTorch version. If not, use Triton-based FP8 matmul kernel.

**Testing:**
- Train 500 steps with FP8 enabled vs disabled
- Compare training loss curves (should be within 1%)
- Measure step time reduction

**Expected: ~22% speedup on forward+backward matmuls -> ~15-20% total speedup**

### Phase 1 Checkpoint

**Combined throughput gain: 20-28% faster -> ~8300-8850 steps in 600s**

Run full 10-min training with GNS+FP8. Measure:
- Total steps completed
- Final BPB (should be -0.002 to -0.004 lower than baseline from more training)
- Training loss curve (should be smooth, no quality degradation)

---

## Phase 2: Loss Function (Approach #4)

### Step 2.1: Implement Multi-Token Prediction Heads

**Files:** `train_gpt.py` GPT class (~line 648)

**Tasks:**
1. Add `mtp_heads` to GPT class: K=3 auxiliary CastedLinear layers (d -> V)
2. Modify `forward()`:
   - Compute main logits from `lm_head` (unchanged)
   - Compute auxiliary logits from `mtp_heads[k]` for k=1,2,3
   - Shift targets: `targets_k = targets[:, k:]` for head k
   - Compute weighted sum of CE losses
3. Add auxiliary head parameters to Adam optimizer (not Muon -- they're 1D projections)
4. At export time: strip `mtp_heads` from state dict

**Key detail:** The auxiliary heads see the same hidden states h_t as the main head. No additional backbone compute.

**Testing:**
- Verify training loss includes all K+1 terms
- Verify auxiliary losses decrease over training (the model is learning to predict future tokens)
- Verify exported model has no extra parameters

### Step 2.2: Implement BPB-Weighted Loss

**Files:** `train_gpt.py` training loop (~line 967)

**Tasks:**
1. Pass `base_bytes_lut` into the training forward pass (currently only used for eval)
2. Compute per-token weight: `w_t = 1.0 / max(1, bytes(target_t))`
3. Normalize weights to mean 1.0 (so the learning rate doesn't need adjustment)
4. Apply weights to CE loss: `loss = (ce * w).mean()`
5. Add warmup: linearly interpolate from uniform to BPB-weighted over first 500 steps

**Testing:**
- Verify BPB improves (even if per-token CE is slightly worse, BPB should be better)
- Verify training stability (no loss spikes from upweighted short tokens)

### Phase 2 Checkpoint

Run full training with Phase 1 + Phase 2. Measure BPB improvement.

**Expected cumulative: -0.005 to -0.012 BPB**

---

## Phase 3: Training Schedule (Approach #2)

### Step 3.1: Implement EphemeralLoRA

**Files:** `train_gpt.py` new class (~30 lines)

**Tasks:**
1. Create `EphemeralLoRA` class with forward hooks on Q, V, Out projections
2. Support `apply()` (register hooks) and `remove()` (deregister hooks)
3. Support `parameters()` for inner loop optimization
4. Support `zero_init()` for resetting between meta-steps

**Testing:**
- Verify hooks add LoRA perturbation to attention output
- Verify removal restores original forward behavior
- Verify gradients flow through LoRA params in inner loop

### Step 3.2: Implement Meta-Warmdown Loop

**Files:** `train_gpt.py` training loop (~line 967)

**Tasks:**
1. When `scale < meta_threshold` (warmdown active):
   a. Split batch into support/query halves
   b. Create EphemeralLoRA, adapt on support with 1 SGD step
   c. Compute query loss with adapted LoRA
   d. Backward through base model (Muon gradient)
   e. Remove LoRA, step optimizer
2. Handle `torch.compile` compatibility (may need to disable dynamo for meta steps)
3. Handle gradient accumulation (meta-step uses half the batch for each phase)

**Critical concern:** The batch split halves the effective batch size for each phase. This may increase gradient noise. Mitigation: increase grad_accum_steps during meta-warmdown.

**Testing:**
- A/B test: train two models (standard warmdown vs meta-warmdown)
- Run identical LoRA TTT on both
- Compare TTT gain: meta-warmdown should show >1.5x improvement

### Phase 3 Checkpoint

Run full training with Phase 1+2+3. Measure TTT gain improvement.

**Expected cumulative: -0.008 to -0.018 BPB**

---

## Phase 4: Architecture Changes (Approaches #1 and #3)

**Note: Approaches #1 and #3 are alternatives, not complements.** Both aim to fit more layers in 16MB, but via different mechanisms:
- #1: Low-rank MLP factors -> 18 layers
- #3: SVD quantized factors -> 13 layers

Implement both, test both, pick the winner. They cannot be combined (you'd be double-compressing the MLP).

### Step 4.1a: Implement Low-Rank Deep Model (Approach #1)

**Files:** `train_gpt.py` MLP class, GPT class, Parallel Muon

**Tasks:**
1. Add `LowRankLinear(in_dim, out_dim, rank)` module
2. Replace MLP `fc` and `proj` with `LowRankLinear`
3. Change `NUM_LAYERS` from 11 to 18
4. Update U-Net skip connections for 18 layers
5. Update parameter banks for factor shapes
6. At GPTQ time: materialize W = A @ B, then quantize

**Testing:**
- Verify parameter count is within 16MB at int6+LZMA
- Verify Muon processes thin factors correctly
- Compare BPB: 18L low-rank vs 11L full-rank

### Step 4.1b: Implement SVD Quantized Factors (Approach #3)

**Files:** `train_gpt.py` QAT section, quantization section

**Tasks:**
1. Add `svd_quantize_ste()` function for SVD-aware fake quantization
2. Replace int6 STE in CastedLinear with SVD-aware STE during QAT
3. Add `quantize_svd_factors()` for post-training export
4. Add `dequantize_svd_factors()` for inference
5. Change `NUM_LAYERS` from 11 to 13 (funded by SVD savings)

**Testing:**
- Measure spectral decay of trained weights (compute E(256) for each layer)
- Compare compression: SVD factors + LZMA vs int6 + LZMA
- Compare BPB: 13L SVD vs 11L int6

### Step 4.2: Pick Winner

Compare:
- Approach #1: 18L low-rank (expected -0.004 to -0.010)
- Approach #3: 13L SVD (expected -0.002 to -0.006)

The winner gets integrated with Phase 1+2+3.

### Phase 4 Checkpoint

Run full stack (Phase 1+2+3+4). Measure final BPB.

**Expected cumulative: -0.012 to -0.025 BPB -> target 1.090-1.103**

---

## Approach Interactions and Conflicts

### Positive Interactions (approaches enhance each other)

| Pair | Interaction |
|------|------------|
| #5 + everything | Faster training benefits all approaches |
| #4 + #1 or #3 | MTP auxiliary gradients are especially valuable for deeper models (more layers to train) |
| #2 + #4 | MTP-trained backbone may adapt better to TTT (richer representations) |

### Negative Interactions (approaches conflict)

| Pair | Conflict | Resolution |
|------|----------|-----------|
| #1 and #3 | Both change architecture for depth; cannot combine | Pick winner |
| #5 FP8 + #2 meta | FP8 precision + meta-gradient noise may compound | Disable FP8 during meta-warmdown phase |
| #4 MTP + #5 FP8 | MTP overhead + FP8 overhead may exceed budget | Run MTP heads in bf16 (small), main model in FP8 |

### Neutral Interactions (orthogonal)

| Pair | Why Independent |
|------|----------------|
| #2 + #5 | Meta-warmdown schedule is independent of training speed |
| #4 + #3 | Loss function is independent of quantization method |

---

## File Modification Summary

All modifications are to `train_gpt.py` (the single-file training script).

| Section | Approaches Touching It | Lines Changed |
|---------|----------------------|---------------|
| Muon optimizer (~96-168) | #5 (GNS) | ~15 |
| CastedLinear (~509-513) | #5 (FP8), #3 (SVD QAT) | ~30 |
| MLP class (~606-617) | #1 (low-rank) | ~15 |
| GPT class (~648-724) | #1 (layers, U-Net), #4 (MTP heads) | ~30 |
| Training loop (~922-1060) | #2 (meta-warmdown), #4 (BPB loss) | ~60 |
| Quantization (~288-422) | #3 (SVD factors) | ~60 |
| Export/eval (~1062-1127) | #4 (strip MTP heads) | ~10 |
| New classes/functions | #1 (LowRankLinear), #2 (EphemeralLoRA), #3 (SVD quant) | ~80 |
| **Total** | | **~300 lines** |

The baseline script is ~1127 lines. With ~300 lines of additions (and ~50 lines removed/replaced), the modified script would be ~1380 lines -- under the 1500-line hard cap.

---

## Testing Strategy (Without 8xH100)

### Local Testing (Apple Silicon / Single GPU)

1. **Unit tests for each approach** (can run on CPU/MPS):
   - GNS: verify output matches NS5
   - FP8: skip (requires H100)
   - LowRankLinear: verify forward matches A@B@x
   - EphemeralLoRA: verify hook registration/removal
   - MTP: verify loss includes all K+1 terms
   - SVD QAT: verify SVD truncation + factor quantization
   - BPB weighting: verify weights sum to T (normalized)

2. **Integration test** (tiny model, 50 steps):
   ```bash
   NUM_LAYERS=3 MODEL_DIM=64 VOCAB_SIZE=32 ITERATIONS=50 \
   TRAIN_BATCH_TOKENS=256 MTP_K=3 BPB_WEIGHTED_LOSS=1 \
   META_WARMDOWN=1 MLP_RANK=16 GNS_ENABLED=1 \
   python train_gpt.py
   ```

3. **Correctness checks**:
   - Training loss decreases (with MTP, total loss includes auxiliary terms)
   - No NaN/Inf in gradients
   - Artifact size within 16MB budget

### Cloud Testing (8xH100)

Each approach requires one 10-min run for validation:

| Experiment | Config | Purpose | Time |
|-----------|--------|---------|------|
| Baseline | Current SOTA | Control | 10 min |
| +GNS+FP8 | Approach #5 only | Measure speedup + BPB | 10 min |
| +MTP+BPB | #5 + #4 | Measure loss function improvement | 10 min |
| +Meta-WD | #5 + #4 + #2 | Measure meta-warmdown TTT gain | 10+10 min (A/B) |
| +LowRank | #5 + #4 + #1 (18L) | Measure depth gain | 10 min |
| +SVD | #5 + #4 + #3 (13L) | Measure SVD gain | 10 min |
| Full stack | Best combo | Final submission | 10 min |

**Total cloud budget: ~80 minutes of 8xH100 time (8 runs)**

---

## Success Criteria

| Milestone | BPB Target | Approaches Active | Confidence |
|-----------|-----------|-------------------|------------|
| Phase 1 done | 1.109-1.113 | #5 (throughput) | HIGH |
| Phase 2 done | 1.104-1.110 | #5 + #4 (loss) | MEDIUM-HIGH |
| Phase 3 done | 1.100-1.107 | #5 + #4 + #2 (meta-warmdown) | MEDIUM |
| Phase 4 done | 1.090-1.103 | #5 + #4 + #2 + best of #1/#3 | MEDIUM |

**Minimum viable submission:** Phase 1 (GNS+FP8 only). If this yields < 1.112 BPB (improving over SOTA 1.1147 after 3-seed validation), submit immediately.

**Target submission:** Full stack at < 1.105 BPB.

---

## Timeline

| Phase | Approaches | Implementation Time | GPU Time |
|-------|-----------|-------------------|----------|
| Phase 1 | #5 GNS+FP8 | 4-6 hours | 20 min |
| Phase 2 | #4 MTP+BPB loss | 4-6 hours | 10 min |
| Phase 3 | #2 Meta-warmdown | 6-8 hours | 20 min |
| Phase 4 | #1 or #3 (architecture) | 6-8 hours each | 20 min |
| Integration | All combined | 2-4 hours | 10 min |
| **Total** | | **22-32 hours** | **80 min** |

**Critical path:** Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 (sequential, each builds on previous).
Phase 4's two architecture approaches (#1 and #3) can be implemented in parallel.
