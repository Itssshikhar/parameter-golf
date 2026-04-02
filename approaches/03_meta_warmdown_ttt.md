# Approach 3: Meta-Learning Warmdown + LoRA TTT

**Expected BPB improvement: -0.004 to -0.012**
**Training cost: ~3% additional**
**Eval cost: ~410s (legal TTT)**
**Risk: MEDIUM**

---

## Problem Statement

25 failed attempts to make TTT work on the current SOTA stack. The model was trained to minimize average CE, not to be adaptable. The loss surface near trained weights has no useful curvature for per-document SGD.

## Core Idea

Replace the warmdown phase (last 3500 steps with diminishing LR returns) with Reptile meta-learning that explicitly shapes the weight landscape for fast LoRA adaptation at test time.

## Mathematical Formulation

### Phase 1: Normal Training (steps 0 to warmdown_start)
Standard Muon+Adam training as current SOTA. No changes.

### Phase 2: Meta-Learning Warmdown (steps warmdown_start to end)

For each meta-step:

```
1. Split batch into D_support and D_query (random 50/50 split)

2. Initialize ephemeral LoRA adapters δ = 0
   (rank-8 on Q, V, Out projections across all 11 layers)
   Total: 225K trainable adapter params

3. Inner loop (LoRA adaptation):
   δ ← δ - α_inner · ∇_δ L(θ + δ, D_support)
   (One SGD step, updating ONLY LoRA params)

4. Outer loop (base model update):
   g_outer = ∇_θ L(θ + δ_adapted, D_query)
   θ ← θ - Muon(g_outer)
   (Standard Muon step on base weights, NOT on LoRA)

5. Discard δ (ephemeral, not stored)
```

### Why This Optimizes Adaptability

The outer loss `L(θ + δ_adapted, D_query)` is minimized w.r.t. θ. This means:

```
θ* = argmin_θ  E[L(θ - α·∇_δ L(θ, D_sup), D_query)]
```

The gradient of this outer objective through the inner loop is:

```
∇_θ L_outer ≈ ∇_θ L(θ + δ, D_query)
            + α · ∇²_θδ L(θ, D_sup) · ∇_δ L(θ + δ, D_query)
```

The first term is the standard gradient. The **second term** is the meta-learning signal — it adjusts θ to make the inner loop gradient (∇_δ) more useful for improving D_query performance. This sculpts the loss landscape to have "adaptation valleys" that LoRA can follow.

### First-Order Approximation (Reptile-style)

To avoid expensive second-order computation:

```
θ_new = θ + β · (θ_after_inner - θ)
```

Where θ_after_inner is the base model state after the inner loop (treating θ as if LoRA updates were applied directly). This is cheaper but less principled.

## Compute Budget Analysis

### Inner Loop Cost
- LoRA params: 225K (rank-8, Q+V+Out, 11 layers)
- Forward through full model: ~12ms (half batch)
- Backward through LoRA only: ~1ms (LoRA is tiny)
- Total inner loop: ~13ms
- As fraction of full step (86ms): **~15%**

### With K=1 Inner Step
- Meta-step time: 86ms (normal) + 13ms (inner) = 99ms
- Overhead: 15%
- But only during warmdown (last ~3500 steps)
- Warmdown as fraction of training: ~50%
- **Net overhead: ~7.5% → reduced to ~3% with careful implementation**

### Training Steps Impact
- Normal: ~6920 steps in 600s
- With meta-warmdown: ~6920 - 3% = ~6710 steps
- Steps lost: ~210
- Each step ≈ 0.001 BPB at convergence → ~0.002 BPB quality cost from fewer steps

### TTT Improvement Needed to Break Even
- Must gain > 0.002 BPB from improved TTT
- Current TTT: -0.0025 BPB (barely breaks even)
- Need: 2-5× improvement → -0.005 to -0.0125 BPB from TTT
- Conservative: 2× → -0.005 BPB → net -0.003 BPB
- Moderate: 3× → -0.0075 BPB → net -0.0055 BPB
- Optimistic: 5× → -0.0125 BPB → net -0.0105 BPB

## Eval-Time LoRA TTT Protocol

```
For each 32K-token chunk (score-first, legal):
  1. Score chunk under torch.inference_mode() (neural model frozen)
  2. Initialize fresh LoRA δ = 0 (or use meta-learned initialization)
  3. For epoch = 1..3:
     For each 2048-token sequence in chunk:
       loss = CE(model(θ + δ, sequence))
       δ ← δ - lr_ttt · ∇_δ loss
  4. Use adapted model (θ + δ) for next chunk's scoring
```

### Key Difference from Standard TTT
- Standard: model θ was trained to minimize E[L(θ)]. LoRA δ adapts from a random direction.
- Meta-learned: model θ was trained to minimize E[L(θ - α∇L(θ))]. LoRA δ adapts along a PREPARED direction.

## LoRA Architecture

```python
class MetaLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

# Applied to Q, V, Out projections in every layer:
# 11 layers × 3 projections × (rank×in + out×rank)
# = 33 × (8×512 + 512×8) = 33 × 8192 = 270K params
# (slightly higher than 225K due to in/out dim differences for K/V)
```

## Falsification Test

Before full implementation, run a quick test:

1. Train the current SOTA normally
2. During warmdown, fork into two branches:
   - Branch A: standard warmdown (current approach)
   - Branch B: meta-learning warmdown (proposed)
3. Run identical LoRA TTT on both
4. Compare TTT gain: if B gets > 2× A's TTT improvement, approach validates

This requires one 8×H100 run (~10 min) per branch = ~20 min total to validate.

## Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Meta-learning doesn't improve TTT | MEDIUM | Approach fails | Falsification test catches this early |
| Inner loop destabilizes Muon | LOW | Training diverges | Start meta-warmdown late (scale < 0.10 instead of 0.15) |
| LoRA rank too low for meaningful adaptation | LOW | Small TTT gain | Test rank 16 if rank 8 is insufficient |
| FineWeb val docs lack enough document-specific structure | MEDIUM | Ceiling on TTT gains | Cannot be mitigated — inherent to task |

## References

- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML.
- Nichol et al. (2018). "On First-Order Meta-Learning Algorithms." arXiv:1803.02999.
- Raghu et al. (2019). "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML." ICLR 2020.
- Parameter Golf PR #549: Legal TTT with SGD, -0.0025 BPB on leader stack.
- Parameter Golf PR #1019: 25 failed TTT attempts documented.
