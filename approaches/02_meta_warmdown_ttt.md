# Approach 2: Reptile Meta-Learning Warmdown for TTT-Optimized Weights

**Type: TRAINING change (optimizer/schedule)**
**Expected BPB improvement: -0.003 to -0.008**
**Training cost: ~10% overhead during warmdown phase only**
**Eval cost: ~410s (legal TTT, same as SOTA)**
**Risk: MEDIUM**

---

## Problem Statement

Test-time training (TTT) is the single largest eval-time gain in the competition (PR #549: -0.0025 BPB). But 25+ attempts to improve TTT beyond that have failed. The root cause: **the model was trained to minimize average cross-entropy, not to be quickly adaptable to new documents.**

The loss surface near the trained weights has curvature that is useful for generalization but not for per-document fine-tuning. A few SGD steps at test time can only exploit the local geometry -- and that geometry was shaped by 7000 steps of average-case optimization, not adaptation-case optimization.

**Solution: Replace the final training phase (warmdown) with Reptile meta-learning that explicitly shapes weights for fast LoRA adaptation at test time.**

This is a training change: it modifies the objective function during the last ~3000 training steps.

---

## Mathematical Formulation

### Phase 1: Standard Training (steps 0 to warmdown_start)

No changes. Standard Muon+Adam training, exactly as current SOTA:

```
theta <- theta - Muon(nabla_theta L(theta, D_batch))
```

### Phase 2: Reptile Meta-Warmdown (steps warmdown_start to end)

At each meta-step:

**Step 1: Split batch into support and query sets**
```
D_batch = {(x_1, y_1), ..., (x_B, y_B)}
D_sup = D_batch[:B/2]     (first half)
D_qry = D_batch[B/2:]     (second half)
```

**Step 2: Initialize ephemeral LoRA adapters**
```
delta = {A_l, B_l} for l in [1..L]   (all layers, Q+V+Out projections)
A_l ~ N(0, 0.01^2),  B_l = 0         (LoRA initialization)
Total adapter params: L * 3 * 2 * r * d = 18 * 3 * 2 * 8 * 512 = 442K
```

Where r=8 is the LoRA rank and d=512 is the model dimension (with appropriate adjustments for Q/K/V dimensions).

**Step 3: Inner loop (K=1 step of LoRA adaptation on support)**
```
g_delta = nabla_delta L(theta + W_lora(delta), D_sup)
delta' = delta - alpha_inner * g_delta
```

Where `W_lora(delta)` merges the LoRA perturbation into the base weights:
```
W_lora(delta)_l = B_l @ A_l     (rank-r perturbation to weight l)
```

**Step 4: Outer loop (Reptile update on base weights)**

The Reptile objective is:
```
theta* = argmin_theta  E_{D_sup, D_qry} [L(theta + W_lora(delta'(theta, D_sup)), D_qry)]
```

The first-order Reptile approximation:
```
theta_after = theta + W_lora(delta')    (base weights + adapted LoRA)
g_reptile = theta - theta_after = -W_lora(delta')
theta <- theta + epsilon * (theta_after - theta)
       = theta + epsilon * W_lora(delta')
```

But we actually want to train the base model with Muon on the query set WHILE the LoRA is applied:
```
g_outer = nabla_theta L(theta + W_lora(delta'), D_qry)
theta <- theta - Muon(g_outer)
```

**Step 5: Discard delta (ephemeral, not stored)**

### Why This Works: The Meta-Gradient

The outer loss is `L(theta + W_lora(delta'(theta, D_sup)), D_qry)` where `delta'` depends on `theta` through the inner loop. The full gradient (via the chain rule through the inner loop) is:

```
dL/d_theta = partial_theta L + partial_delta' L * d_delta'/d_theta
```

The second term is the **meta-learning signal**. It pushes theta toward regions where:
1. The inner loop gradient on D_sup is informative for D_qry
2. Small LoRA perturbations produce large query improvements
3. The weight space has "adaptation valleys" that LoRA can follow

With the first-order (Reptile) approximation, we avoid computing the expensive second-order term explicitly. Instead, the Muon update on the query loss with LoRA applied achieves a similar effect: it moves theta in directions that are good *after* LoRA adaptation.

### Connection to MAML/Reptile Theory

Nichol et al. (2018) proved that Reptile's expected update direction contains:

```
E[g_reptile] = E[nabla L(theta)] + alpha * E[nabla^2 L * nabla L] + O(alpha^2)
```

The second term maximizes the inner product between gradients on different mini-batches. This is exactly the "adaptability" signal: weights where the gradient from one document (support) aligns with the useful direction for another document (query).

---

## How It Builds on the SOTA Stack

| Component | Change? | Details |
|-----------|---------|---------|
| Training steps 0 to warmdown_start | No change | Standard Muon+Adam |
| Warmdown training loop | **MODIFIED** | Reptile meta-learning replaces standard warmdown |
| Muon optimizer | No change | Used for outer loop update |
| Adam (embeddings) | No change | Same |
| EMA (0.997) | No change | Continues during meta-warmdown |
| SWA | No change | Checkpoints collected same as before |
| Late QAT | **Compatible** | Run QAT inside the outer loop (after LoRA removal) |
| Architecture (11L, 512d, etc.) | No change | Same model |
| GPTQ + LZMA | No change | Post-training |
| TTT at eval time | **ENHANCED** | The whole point -- TTT gains should increase 2-5x |

### Integration with Legal Score-First TTT

At eval time, the protocol remains identical to current SOTA:
```
For each 32K-token chunk (score-first, legal):
  1. Score chunk under torch.inference_mode()
  2. Train LoRA adapters on scored tokens (SGD, lr=0.002, 3 epochs)
  3. Use adapted model for next chunk's scoring
```

The difference is that the base weights theta are now **meta-optimized** for this exact adaptation protocol. The LoRA updates at test time will be more effective because:
- The loss landscape around theta has been shaped for fast adaptation
- The LoRA subspace (Q/V/Out projections) aligns with the meta-learned adaptation directions
- A few SGD steps can exploit the pre-shaped curvature

---

## Compute Budget Analysis

### Per-Step Overhead During Warmdown

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Normal forward on D_sup | ~43 | Half batch |
| LoRA forward (tiny) | ~1 | 442K params, trivial |
| LoRA backward | ~2 | Only through LoRA params |
| Forward on D_qry with LoRA | ~43 | Half batch |
| Backward on D_qry (full model) | ~43 | Standard Muon gradient |
| Muon step | ~5 | Same as normal |
| **Total meta-step** | **~137** | vs ~86ms normal step |

Overhead: ~60% per step during warmdown only.

### Impact on Total Training

```
Warmdown fraction: 3000/6920 = 43% of training
Steps before warmdown: 3920 steps * 86ms = 337s
Remaining time: 600 - 337 = 263s
Meta-steps in remaining time: 263s / 137ms = ~1920 steps (vs ~3000 normal warmdown steps)
Steps lost: ~1080 warmdown steps

Total steps: 3920 + 1920 = 5840 (vs 6920 baseline = -16%)
```

### Break-Even Analysis

Each step at convergence contributes ~0.0001 BPB. Losing 1080 steps costs ~0.001 BPB.

For the approach to be net positive, TTT must improve by more than 0.001 BPB beyond its current -0.0025 BPB gain.

Current TTT: -0.0025 BPB
Required TTT with meta-warmdown: > -0.0035 BPB (i.e., >40% improvement)

**Conservative (2x current TTT): -0.005 BPB -> net +0.0015 BPB over baseline TTT, minus 0.001 step cost = net -0.003 BPB**
**Moderate (3x current TTT): -0.0075 BPB -> net +0.004 over baseline TTT = net -0.005 BPB**
**Optimistic (5x current TTT): -0.0125 BPB -> net +0.009 over baseline = net -0.008 BPB**

The literature on MAML/Reptile typically shows 2-5x improvement in few-shot adaptation accuracy. In the language modeling context, a 2-3x improvement in TTT gain is the conservative expectation.

---

## Implementation Plan

### Files to Modify: `train_gpt.py`

1. **Add `EphemeralLoRA` class** (~30 lines):
```python
class EphemeralLoRA:
    """Lightweight LoRA adapters that are created/destroyed per meta-step."""
    def __init__(self, model, rank=8):
        self.hooks = []
        self.params = []
        for name, mod in model.named_modules():
            if isinstance(mod, CastedLinear) and any(k in name for k in ['c_q', 'c_v', 'c_proj']):
                in_d, out_d = mod.weight.shape[1], mod.weight.shape[0]
                A = torch.randn(rank, in_d, device=mod.weight.device, dtype=torch.bfloat16) * 0.01
                B = torch.zeros(out_d, rank, device=mod.weight.device, dtype=torch.bfloat16)
                A.requires_grad_(True)
                B.requires_grad_(True)
                self.params.extend([A, B])
                # Register forward hook to add B @ A @ x to output
                hook = mod.register_forward_hook(
                    lambda m, inp, out, A=A, B=B: out + F.linear(F.linear(inp[0], A), B)
                )
                self.hooks.append(hook)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
```

2. **Modify warmdown training loop** (~40 lines):
```python
# In main training loop, when warmdown is active:
if meta_warmdown_enabled and scale < meta_threshold:
    # Split batch
    x_sup, y_sup = x[:B//2], y[:B//2]
    x_qry, y_qry = x[B//2:], y[B//2:]

    # Inner loop: create LoRA, adapt on support
    lora = EphemeralLoRA(base_model, rank=8)
    loss_sup = model(x_sup, y_sup)
    lora_grads = torch.autograd.grad(loss_sup, lora.params, retain_graph=False)
    with torch.no_grad():
        for p, g in zip(lora.params, lora_grads):
            p.sub_(alpha_inner * g)

    # Outer loop: compute query loss with adapted LoRA
    optimizer.zero_grad()
    loss_qry = model(x_qry, y_qry)
    loss_qry.backward()

    # Remove LoRA before optimizer step
    lora.remove()
    del lora

    # Standard Muon/Adam step on base model
    for opt in optimizers:
        opt.step()
else:
    # Standard training step (unchanged)
    ...
```

3. **Add env vars** (~5 lines):
   - `META_WARMDOWN=1`
   - `META_THRESHOLD=0.15` (activate when LR scale < 0.15, same as late QAT)
   - `META_LORA_RANK=8`
   - `META_ALPHA_INNER=0.01`

**Total: ~75-90 lines modified/added**

### Falsification Test

Before full implementation, run a quick A/B test:

1. Train current SOTA normally (Branch A)
2. Train with meta-warmdown (Branch B)
3. Run identical LoRA TTT on both
4. Compare TTT gain: if B > 1.5x A, approach validates

Cost: 2 x 10-min runs = 20 min on 8xH100.

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Meta-warmdown doesn't improve TTT gain | MEDIUM | Wasted warmdown steps | Falsification test in 20 min catches this early |
| Inner loop destabilizes Muon convergence | LOW | Training diverges | Start meta later (scale < 0.10), use smaller alpha_inner |
| LoRA rank too low for meaningful adaptation | LOW | Ceiling on TTT gain | Test rank=16 or rank=32 |
| FineWeb val docs lack document-specific patterns | MEDIUM | TTT ceiling is low regardless | Fundamental limitation -- can't mitigate |
| torch.compile breaks with dynamic hooks | MEDIUM | Fallback to eager mode during warmdown | Use `torch._dynamo.disable()` context manager for meta steps |
| Half-batch reduces effective batch size | LOW | Noisier gradients | Meta-learning benefits from batch diversity (feature, not bug) |

---

## What Makes It NOVEL

**Not done by anyone on the leaderboard:**
- All TTT submissions (PR #549, PR #1019 "25 failed attempts") use standard-trained models
- No submission has used meta-learning during training
- The warmdown phase (last 3000-3500 steps) is currently just LR decay -- this replaces it with a fundamentally different optimization objective

**Distinct from standard warmdown:**
- Standard warmdown: minimize E[L(theta)] with diminishing LR
- Meta-warmdown: minimize E[L(theta + delta'(theta))] -- optimize for post-adaptation quality

**Distinct from fine-tuning tricks:**
- This changes the pretraining loss function during warmdown, not the fine-tuning recipe
- The model ships with the same architecture and parameter count
- Only the weight geometry is different (shaped for adaptation)

---

## References

- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
- Nichol et al. (2018). "On First-Order Meta-Learning Algorithms." arXiv:1803.02999.
- Raghu et al. (2019). "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML." ICLR 2020.
- Sun et al. (2020). "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts." ICML 2020.
- Gandelsman et al. (2022). "Test-Time Training with Masked Autoencoders." NeurIPS 2022.
- Parameter Golf PR #549: Legal TTT with SGD, -0.0025 BPB.
- Parameter Golf PR #1019: 25 failed TTT attempts (all with standard-trained models).
