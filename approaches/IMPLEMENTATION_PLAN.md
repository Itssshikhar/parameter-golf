# Implementation Plan — All 5 Approaches

## Decision: Single Plan vs Separate Plans

**Single plan is better** because the approaches share the same codebase (`train_gpt.py`), interact with each other, and should be implemented in a specific order to enable incremental testing. Each approach modifies a different part of the pipeline, so merge conflicts are minimal.

---

## Implementation Order (by risk and dependency)

```
Phase 1: EVAL-ONLY changes (zero training risk)
  ├── Approach #5: Adaptive Stride          [2-3 hours, zero risk]
  └── Approach #1: EWMA-gram               [3-4 hours, low risk]

Phase 2: POST-TRAINING changes (zero training risk)
  ├── Approach #2A: Klein-GPTQ             [2-3 hours, zero risk]
  └── Approach #2B: Learned Interpolation   [2-3 hours, zero risk]

Phase 3: TRAINING changes (requires retraining)
  ├── Approach #2C: SAQ                     [2-3 hours, low risk]
  ├── Approach #3: Meta-Warmdown TTT        [4-6 hours, medium risk]
  └── Approach #4: Low-Rank Deep Model      [6-8 hours, high risk]
```

This order ensures: (1) we get immediate measurable results from eval-only changes, (2) we validate the eval infrastructure before making training changes, (3) the riskiest changes come last.

---

## Phase 1: Eval-Only Changes

### Step 1.1: Fork the SOTA train_gpt.py

```
Source: records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py
Target: parameter-golf-bunny/train_gpt.py
```

- Copy the current SOTA as our starting point
- Verify it runs and reproduces ~1.1147 BPB (need 8×H100 access)
- All subsequent modifications are relative to this file

### Step 1.2: Implement Adaptive Stride (Approach #5)

**Files modified:** `train_gpt.py` — eval section only

**Tasks:**
1. Add a new function `eval_val_adaptive_stride()` alongside existing `eval_val_sliding()`
2. Phase 1 (coarse): call existing sliding window logic with stride=256, collect per-position NLLs
3. Phase 2 (classify): compute NLL quantiles (25th, 75th percentile)
4. Phase 3-4 (fine/standard): re-call sliding window on position subsets with stride=16 and stride=64
5. Phase 5 (combine): merge scores, compute final BPB
6. Add env vars: `ADAPTIVE_STRIDE=1`, `STRIDE_FINE=16`, `STRIDE_COARSE=256`, `HARD_FRACTION=0.25`
7. Wire into the final eval section (after quantization roundtrip)

**Key implementation detail:** The existing `eval_val_sliding` processes windows sequentially per rank. For position-selective evaluation, we need to generate window schedules that cover only the required positions. Simplest approach: generate all windows for stride=16, then filter to only those whose scored positions overlap with the hard set.

**Testing (without 8×H100):**
- Can test the coarse→classify→selective logic on CPU with a tiny model
- Cannot test actual BPB improvement without the SOTA model + validation data on GPU

**Estimated lines:** ~60-80 new lines

### Step 1.3: Implement EWMA-gram (Approach #1)

**Files modified:** `train_gpt.py` — eval section only

**Tasks:**
1. Add EWMA counter state initialization (uniform prior)
2. Modify `eval_val_sliding` (or the adaptive variant) to:
   a. After computing logits, compute P_neural via softmax
   b. Look up P_bi and P_uni from counters
   c. Compute entropy H of P_neural
   d. Compute entropy-adaptive mixing weight α
   e. Compute P_final = (1-α_bi-α_uni)·P_neural + α_bi·P_bi + α_uni·P_uni
   f. Compute NLL from P_final (not from raw CE loss)
   g. Update counters with the scored token (AFTER scoring)
3. Handle the model forward pass to return logits instead of loss (need `forward_logits` or modify forward)
4. Add env vars: `EWMA_ENABLED=1`, `EWMA_LAMBDA_UNI=0.999`, `EWMA_LAMBDA_BI=0.995`, `EWMA_ALPHA_BI_MAX=0.10`, `EWMA_ALPHA_UNI_MAX=0.03`, `EWMA_GATE_W=2.0`, `EWMA_GATE_B=-3.0`
5. Vectorize: process all scored positions in a window as a batch (avoid per-token Python loop)

**Critical implementation consideration:**
The current model's `forward()` returns `F.cross_entropy(logits, targets)` — a scalar loss. EWMA-gram needs the raw logits to compute P_neural. Options:
- Add a `forward_logits()` method that returns logits without computing loss
- Or split the existing forward into logits + loss steps

The SOTA code already has a `forward_logits` or similar for the GPTQ calibration data generation (line ~1081 in the ValCalib submission). Reuse that.

**Estimated lines:** ~70-90 new/modified lines

### Phase 1 Checkpoint

After implementing 1.2 and 1.3, we have:
- Adaptive stride evaluation
- EWMA-gram mixing

Both are eval-only. Test by running the SOTA model with these eval modifications on 8×H100.

**Expected improvement from Phase 1: -0.006 to -0.017 BPB**

---

## Phase 2: Post-Training Changes

### Step 2.1: Implement Klein-Randomized GPTQ (Approach #2A)

**Files modified:** `train_gpt.py` — quantization section

**Tasks:**
1. Modify the existing `quantize_int6_gptq` function (or add `quantize_int6_klein`)
2. Add outer loop: `for trial in range(K):`
3. Inside each trial:
   a. Generate random column permutation
   b. Reorder H and W columns accordingly
   c. Recompute Cholesky on permuted H
   d. Add stochastic noise to rounding: `q = round(w/s + noise)` where noise ~ N(0, temperature)
   e. Run standard GPTQ column-by-column with error compensation
   f. Compute Hessian-weighted MSE for this trial
4. Keep the trial with lowest MSE
5. Add env vars: `KLEIN_TRIALS=32`, `KLEIN_TEMPERATURE=0.1`

**Key detail:** The existing GPTQ code already has the Cholesky computation. The modification is wrapping it in a loop with random permutations and stochastic rounding.

**Estimated lines:** ~30 modified/new lines

### Step 2.2: Implement Learned Checkpoint Interpolation (Approach #2B)

**Files modified:** `train_gpt.py` — post-training section

**Tasks:**
1. During warmdown, save K=15 checkpoints to CPU memory (modify SWA collection code):
   ```python
   if swa_enabled and step % swa_every == 0 and scale < swa_start:
       checkpoints.append({name: t.detach().cpu().clone() for name, t in model.state_dict().items()})
   ```
2. After training completes, before quantization:
   a. Define `eval_mix(alpha_logits)` function that:
      - Computes α = softmax(logits)
      - Mixes checkpoints: w_mix = Σ α_k · w_k
      - Loads into model, runs fast eval on 1% of val tokens
      - Returns BPB
   b. Run Nelder-Mead optimization (from scipy or manual implementation)
   c. Apply optimal α to get final weights
3. Add env vars: `TWA_ENABLED=1`, `TWA_EVAL_FRACTION=0.01`, `TWA_MAX_ITER=80`

**Dependency note:** scipy may not be available in the RunPod environment. Implement Nelder-Mead manually (~40 lines) or use `torch.optim` with numerical gradients.

**Estimated lines:** ~50-60 new lines

### Phase 2 Checkpoint

After Phase 2, we have all eval-only and post-training improvements. Test end-to-end on 8×H100.

**Expected cumulative improvement from Phase 1+2: -0.010 to -0.024 BPB**

---

## Phase 3: Training Changes

### Step 3.1: Implement SAQ (Approach #2C)

**Files modified:** `train_gpt.py` — CastedLinear class, training loop

**Tasks:**
1. Modify `CastedLinear.forward()` when QAT is enabled:
   ```python
   if self._qat_enabled and self.training:
       # Standard fake quantize
       w_q = fake_quantize_int6(w)
       loss = F.linear(x, w_q)

       # SAQ: compute perturbation direction
       # (requires loss.backward() first, then ascent step)
       # This is tricky because we're inside the forward pass
   ```

   **Better approach:** Implement SAQ at the training loop level, not inside CastedLinear:
   ```python
   # In training loop, when late QAT is active:
   # Step 1: normal forward + backward
   loss = model(x, y)
   loss.backward()

   # Step 2: compute perturbation
   with torch.no_grad():
       for p in qat_params:
           epsilon = rho * (p.grad / (p.grad.norm() + 1e-8))
           p.data.add_(epsilon)

   # Step 3: perturbed forward + backward (this replaces the gradient)
   optimizer.zero_grad()
   loss_saq = model(x, y)
   loss_saq.backward()

   # Step 4: undo perturbation
   with torch.no_grad():
       for p in qat_params:
           p.data.sub_(epsilon)

   # Step 5: optimizer step with SAQ gradient
   optimizer.step()
   ```

2. Add env vars: `SAQ_ENABLED=1`, `SAQ_RHO=0.03`
3. Only activate when `late_qat_threshold` is triggered (last ~520 steps)

**Training cost:** 2× compute during the last ~520 steps = ~7.5% overhead = ~520 fewer total steps

**Estimated lines:** ~25 new lines in training loop

### Step 3.2: Implement Meta-Warmdown (Approach #3)

**Files modified:** `train_gpt.py` — training loop, new LoRA module, eval loop

**Tasks:**
1. Add `EphemeralLoRA` module:
   ```python
   class EphemeralLoRA:
       def __init__(self, model, rank=8, target_modules=['c_q', 'c_v', 'proj']):
           self.adapters = {}  # name → (A, B) pairs
           for name, module in model.named_modules():
               if any(t in name for t in target_modules) and hasattr(module, 'weight'):
                   in_dim, out_dim = module.weight.shape[1], module.weight.shape[0]
                   A = torch.randn(rank, in_dim, device=module.weight.device) * 0.01
                   B = torch.zeros(out_dim, rank, device=module.weight.device)
                   self.adapters[name] = (nn.Parameter(A), nn.Parameter(B))

       def apply(self):
           # Add LoRA to model forward hooks

       def remove(self):
           # Remove hooks

       def parameters(self):
           return [p for a, b in self.adapters.values() for p in [a, b]]

       def zero_init(self):
           for a, b in self.adapters.values():
               nn.init.normal_(a, std=0.01)
               nn.init.zeros_(b)
   ```

2. Modify warmdown training loop:
   ```python
   if meta_warmdown_enabled and scale < meta_threshold:
       # Split batch
       x_sup, y_sup = x[:B//2], y[:B//2]
       x_qry, y_qry = x[B//2:], y[B//2:]

       # Inner loop: adapt LoRA on support set
       lora = EphemeralLoRA(base_model, rank=8)
       lora.apply()
       loss_sup = model(x_sup, y_sup)
       lora_grad = torch.autograd.grad(loss_sup, lora.parameters())
       with torch.no_grad():
           for p, g in zip(lora.parameters(), lora_grad):
               p.sub_(alpha_inner * g)

       # Outer loop: compute query loss with adapted LoRA
       loss_qry = model(x_qry, y_qry)
       loss_qry.backward()  # gradients flow to base model θ, NOT to LoRA

       lora.remove()
       # Normal optimizer step on base model
       for opt in optimizers:
           opt.step()
   ```

3. Modify eval TTT:
   ```python
   # Same LoRA architecture, but now the base model responds better to LoRA adaptation
   # because it was meta-trained for this
   ```

4. Add env vars: `META_WARMDOWN=1`, `META_THRESHOLD=0.15`, `META_LORA_RANK=8`, `META_ALPHA_INNER=0.01`

**Estimated lines:** ~80-120 new lines

### Step 3.3: Implement Low-Rank Deep Model (Approach #4)

**Files modified:** `train_gpt.py` — model architecture (GPT class, MLP class, parameter banks)

**Tasks:**
1. Add `LowRankLinear` module (see approach doc)
2. Add `LowRankMLP` module that uses `LowRankLinear` for fc and proj
3. Modify GPT class:
   - Change `NUM_LAYERS` from 11 to 18
   - Replace MLP construction with LowRankMLP(rank=128)
   - Update U-Net skip connections for 18 layers (9 encoder + 9 decoder)
   - Update parameter banking for 18 layers
4. Modify Parallel Muon to handle the thin factor banks:
   - `mlp_up_A_bank`: (18, 512, 128)
   - `mlp_up_B_bank`: (18, 128, 1536)
   - `mlp_down_A_bank`: (18, 1536, 128)
   - `mlp_down_B_bank`: (18, 128, 512)
5. Update quantization to handle low-rank factors:
   - Option A: quantize A and B separately (int6 each)
   - Option B: materialize W=A·B, then quantize W with GPTQ
   - Option B is better (GPTQ sees the actual weight distribution)
6. Add env vars: `MLP_RANK=128`, `NUM_LAYERS=18`

**Testing strategy:**
- First test 13 layers with rank=192 (conservative)
- If it works, push to 18 layers with rank=128
- Monitor: pre-quant BPB, post-quant BPB, artifact size, step time

**Estimated lines:** ~60-80 modified lines (replacing MLP class, updating GPT init and forward)

### Phase 3 Checkpoint

After Phase 3, we have the full stack. Run complete experiments on 8×H100:
- Baseline: current SOTA (1.1147)
- + Phase 1 only (eval changes)
- + Phase 1+2 (eval + post-training)
- + Phase 1+2+3 (everything)

**Expected cumulative improvement: -0.015 to -0.035 BPB → target BPB: 1.08-1.10**

---

## Testing Without 8×H100

For local development/testing on Apple Silicon or smaller GPUs:

1. **Unit tests**: Each approach can be tested independently on a tiny model (2 layers, dim=64, vocab=32)
2. **Integration test**: Run the full pipeline on the MLX baseline with `ITERATIONS=50 TRAIN_BATCH_TOKENS=256`
3. **Correctness checks**:
   - EWMA-gram: verify P_final sums to 1, counters update correctly
   - Klein-GPTQ: verify MSE decreases with more trials
   - Adaptive stride: verify token classification matches NLL distribution
   - SAQ: verify perturbation direction is correct (should increase loss)
   - Low-rank: verify LowRankLinear(512, 1536, 128) output matches (A@B)@x

---

## File Organization

```
parameter-golf-bunny/
├── approaches/
│   ├── 01_ewma_gram.md
│   ├── 02_geometry_pipeline.md
│   ├── 03_meta_warmdown_ttt.md
│   ├── 04_lowrank_deep_model.md
│   ├── 05_adaptive_stride.md
│   └── IMPLEMENTATION_PLAN.md      ← this file
├── code_analysis.md
├── done.md
└── train_gpt.py                    ← our modified version (on bunny/work branch)
```

---

## Timeline Estimate

| Phase | Approaches | Time | Requires GPU? |
|-------|-----------|------|---------------|
| Phase 1 | #5 + #1 (eval-only) | 5-7 hours | Only for BPB measurement |
| Phase 2 | #2A + #2B (post-training) | 4-6 hours | Only for BPP measurement |
| Phase 3a | #2C (SAQ) | 2-3 hours | Yes (retraining) |
| Phase 3b | #3 (Meta-Warmdown) | 4-6 hours | Yes (retraining) |
| Phase 3c | #4 (Low-Rank Deep) | 6-8 hours | Yes (retraining) |
| **Total** | **All 5** | **21-30 hours** | |

**Critical path**: Phase 1+2 can be implemented locally. Phase 3 requires 8×H100 access for validation.

---

## Success Criteria

| Milestone | BPB Target | Approaches Active |
|-----------|-----------|-------------------|
| Phase 1 done | 1.107-1.112 | #5 + #1 |
| Phase 2 done | 1.102-1.108 | #5 + #1 + #2A + #2B |
| SAQ added | 1.098-1.105 | + #2C |
| Meta-TTT added | 1.093-1.100 | + #3 |
| Low-Rank Deep | 1.080-1.095 | + #4 |

**Minimum viable submission**: Phase 1+2 (eval+post-training changes only). If this yields < 1.110 BPB (beating current SOTA of 1.1147), submit immediately.
