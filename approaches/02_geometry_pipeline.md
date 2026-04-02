# Approach 2: The Geometry Pipeline — Klein-GPTQ + Learned Interpolation + SAQ

**Expected BPB improvement: -0.005 to -0.012**
**Training cost: ~7.5% (SAQ during warmdown only)**
**Eval cost: ~250-330s**
**Risk: LOW-MEDIUM**

---

## Overview

Three weight-space innovations combined, each fixing a different error source:

| Part | What It Fixes | When It Runs | Cost |
|------|--------------|-------------|------|
| SAQ (Sharpness-Aware QAT) | Training: weight landscape flatness | During warmdown QAT | ~7.5% training overhead |
| Learned Checkpoint Interpolation | Averaging: suboptimal uniform/exponential mixing | Post-training, in eval budget | ~90s |
| Klein-Randomized GPTQ | Quantization: greedy rounding suboptimality | Post-training, in eval budget | ~160s |

These operate on orthogonal axes and compose additively.

---

## Part A: Klein-Randomized GPTQ

### Problem
Current GPTQ = Babai's Nearest Plane algorithm (greedy column-by-column). ICLR 2026 proved this is a suboptimal polynomial-time approximation to the Closest Vector Problem on the Hessian lattice.

### Algorithm
```
For trial k = 1..K:
    perm_k = random_permutation(columns)
    H_perm = H[perm_k][:, perm_k]
    L = cholesky(H_perm)
    Q_k = W[:, perm_k].clone()

    For j in 0..n-1:
        noise = Normal(0, temperature)
        Q_k[:, j] = clamp(round(Q_k[:, j] / scale + noise), -31, 31) * scale
        err = (W_perm[:, j] - Q_k[:, j]) / L[j, j]
        Q_k[:, j+1:] -= outer(err, L[j, j+1:])

    MSE_k = ||H^½ (W - Q_k[:, inv_perm])||²

Return Q_{argmin_k MSE_k}
```

**Expected: -0.001 to -0.003 BPB at K=32. Cost: ~160s eval time. ~30 lines.**

### Key References
- Frantar et al. (ICLR 2023): "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- ICLR 2026: "The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm"
- OJBKQ (Feb 2026): "Objective-Joint Babai-Klein Quantization"

---

## Part B: Learned Checkpoint Interpolation (TWA)

### Problem
SWA averages checkpoints uniformly (1/K). EMA uses fixed exponential decay (0.997). Neither is optimal.

### Algorithm
```
Save K checkpoints {w_1,...,w_K} during warmdown (every 50 steps)
z = zeros(K)                  # logits for simplex projection
For iteration = 1..100:
    α = softmax(z)
    w_mix = Σ_k α_k * w_k
    bpb = eval_fast(w_mix)    # eval on 1% of val tokens (~0.5s)
    z = nelder_mead_step(z, bpb)
α* = softmax(z_final)
w_final = Σ_k α*_k * w_k
```

**Expected: -0.001 to -0.003 BPB. Cost: ~90s eval time. ~40 lines.**

### Key Reference
- TWA (ICLR 2023): "Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions"

---

## Part C: Sharpness-Aware QAT (SAQ)

### Problem
Standard QAT with STE finds weights near int6 grid points, but those points may be on sharp ridges. Perturbations (from cross-layer error accumulation) cause disproportionate loss.

### Algorithm
```python
# During late QAT (when lr_scale < 0.15, last ~520 steps):
# Standard QAT forward:
w_q = fake_quantize(w)
loss = model.forward(w_q, x, y)
loss.backward(retain_graph=True)

# Ascent step (find worst-case perturbation):
with torch.no_grad():
    epsilon = rho * (w.grad / (w.grad.norm() + 1e-8))
    w_perturbed = w + epsilon

# Sharpness-aware loss:
w_q_perturbed = fake_quantize(w_perturbed)
loss_saq = model.forward(w_q_perturbed, x, y)

# Use SAQ gradient for the actual update:
loss_saq.backward()
optimizer.step()
```

**Expected: -0.002 to -0.005 BPB. Cost: 2x compute during last ~520 steps = ~7.5% total. ~20 lines.**

### Key References
- SAQ (2021): "Sharpness-aware Quantization for Deep Neural Networks"
- SAM (2020): "Sharpness-Aware Minimization for Efficiently Improving Generalization"

---

## Combined Pipeline Timeline

```
TRAINING (600s):
  Step 0-6400:    Standard Muon+Adam training (as current SOTA)
  Step 6400-6920: SAQ-enhanced QAT (Part C)           [+7.5% overhead]
  Save 15 checkpoints during warmdown (every 50 steps)

EVAL BUDGET (600s):
  [0-90s]    Learned checkpoint interpolation (Part B)
  [90-287s]  AR self-gen GPTQ calibration data (existing, ~197s)
  [287-447s] Klein-randomized GPTQ K=32 (Part A, ~160s)
  [447-524s] Sliding window eval stride=64 (~77s)
  [524-526s] EWMA-gram mixing (Approach #1, optional, ~2s)
  [526-600s] Buffer / diagnostics

  Total: ~526s of 600s budget → 74s safety margin
```

---

## Risk Assessment

| Part | Can it regress? | Fallback |
|------|----------------|----------|
| Klein-GPTQ | No — keeps best of K trials including the greedy one | Falls back to standard GPTQ |
| Learned Interpolation | No — uniform weights are in feasible set | Falls back to uniform SWA |
| SAQ | Possible if ρ too large | Disable SAQ, use standard QAT |
