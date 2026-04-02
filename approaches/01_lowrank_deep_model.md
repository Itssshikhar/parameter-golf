# Approach 1: Low-Rank Factored MLP Training for Deeper Models

**Type: ARCHITECTURE change (train-time, not post-hoc)**
**Expected BPB improvement: -0.004 to -0.010**
**Artifact cost: Net zero (savings fund more layers)**
**Training cost: Faster per-step (smaller matmuls)**
**Risk: MEDIUM**

---

## Problem Statement

The 16MB artifact limit caps the SOTA at 11 layers (512d, 3x MLP, int6+LZMA). MLP weights consume ~67% of total parameters (17.3M of 26M). The depth-BPB relationship from empirical data shows:

| Layers | BPB   | Delta    |
|--------|-------|----------|
| 9      | 1.1458| baseline |
| 10     | 1.1428| -0.0030  |
| 11     | 1.1271| -0.0025  |

Each additional layer yields ~0.003 BPB improvement with diminishing returns. If we can fit 15-18 layers in 16MB by compressing MLPs during training, the depth gains outweigh the per-layer quality loss from rank reduction.

**This is an architecture change, not an eval trick.** The model is trained from scratch with factored weights.

---

## Mathematical Formulation

### Standard MLP (current SOTA)

```
h = LeakyReLU(0.5)(W_up * x)^2    W_up: [512, 1536]   (786,432 params)
y = W_down * h                      W_down: [1536, 512]  (786,432 params)
Total per layer: 1,572,864 params
Total 11 layers MLP: 17,301,504 params
```

### Low-Rank Factored MLP

Replace each MLP weight with two trained factors:

```
W_up  = B_up * A_up     A_up: [512, r],  B_up: [r, 1536]
W_down = B_down * A_down  A_down: [1536, r], B_down: [r, 512]

Per-layer MLP params: 2 * r * (512 + 1536) = 4096r
```

The forward pass computes:
```
h = LeakyReLU(0.5)((B_up @ (A_up @ x)))^2
y = B_down @ (A_down @ h)
```

### Parameter Budget Analysis

Non-MLP params per layer (attention Q/K/V/O + controls + norms): ~787K + ~2K = ~789K

| Config          | r   | MLP/layer | Attn/layer | Total/layer | Layers | Total params | At int6+LZMA |
|-----------------|-----|-----------|------------|-------------|--------|-------------|-------------|
| SOTA (full)     | 512 | 1,573K    | 789K       | 2,362K      | 11     | 26.5M       | ~15.9 MB    |
| Conservative    | 192 | 787K      | 789K       | 1,576K      | 15     | 24.1M       | ~15.5 MB    |
| **Proposed**    | 128 | 524K      | 789K       | 1,313K      | 18     | 24.3M       | ~15.8 MB    |
| Aggressive      | 96  | 393K      | 789K       | 1,182K      | 20     | 24.1M       | ~15.6 MB    |

**Recommended: r=128, 18 layers.** This gives 7 additional layers at ~75% of the per-layer MLP capacity.

### Spectral Energy Analysis

For a trained weight matrix W with singular values sigma_1 >= ... >= sigma_min(m,n), the rank-k approximation quality is:

```
E(k) = sum_{i=1}^{k} sigma_i^2 / sum_{i=1}^{min(m,n)} sigma_i^2
```

Key insight: **trained low-rank is NOT the same as post-hoc SVD truncation.** When we train with rank-128 factors from scratch:

1. The model never develops patterns requiring rank > 128
2. The optimization landscape is restricted to the rank-128 manifold from step 0
3. The model redistributes its capacity, using all 128 dimensions effectively

Evidence from the literature:
- Aghajanyan et al. (2021): Language models' intrinsic dimensionality is much lower than parameter count
- Zhao et al. (GaLore, 2024): Low-rank gradient projection achieves comparable quality at 30-50% parameter savings
- Hu et al. (LoRA, 2021): Rank-8 adaptation matches full fine-tuning, suggesting rank redundancy in pretrained weights

---

## How It Builds on the SOTA Stack

**Everything stays the same except MLP weight shapes:**

| Component | Change? | Details |
|-----------|---------|---------|
| XSA (all layers) | Extend | Now applied to all 18 layers (or last 6) |
| BigramHash 3072 | No change | Independent of model depth |
| Parallel Muon | Adapt | Factor banks instead of full weight banks |
| LeakyReLU(0.5)^2 | No change | Same activation |
| GQA (8/4 heads) | No change | Same attention config |
| Partial RoPE | No change | Same 16/64 dims |
| U-Net skips | Adapt | 9 encoder + 9 decoder (was 5+6) |
| EMA + SWA | No change | Same averaging |
| Late QAT (int6 STE) | No change | Quantize the factors A, B directly |
| AR Self-Gen GPTQ | Adapt | GPTQ on factors or materialized product |
| LZMA compression | No change | Same compression |

### Muon Compatibility

Muon's Newton-Schulz orthogonalization operates on 2D matrices. The factors A_up (512x128) and B_up (128x1536) are standard 2D matrices -- fully compatible. The Muon scale correction `max(1, rows/cols)^0.5` handles rectangular shapes:

```
A_up  (512 x 128):  scale = sqrt(512/128) = 2.0
B_up  (128 x 1536): scale = 1.0 (128 < 1536, so max(1, 128/1536)^0.5 = 1.0)
A_down (1536 x 128): scale = sqrt(1536/128) = 3.46
B_down (128 x 512):  scale = 1.0
```

### Parameter Banking (Parallel Muon)

Replace single MLP banks with factor banks:

```python
# Current SOTA:
self.mlp_up_bank = nn.Parameter(torch.empty(11, 1536, 512))     # 8.65M params
self.mlp_down_bank = nn.Parameter(torch.empty(11, 512, 1536))   # 8.65M params

# Low-rank:
self.mlp_up_A_bank = nn.Parameter(torch.empty(18, 128, 512))    # 1.18M params
self.mlp_up_B_bank = nn.Parameter(torch.empty(18, 1536, 128))   # 3.54M params
self.mlp_down_A_bank = nn.Parameter(torch.empty(18, 128, 1536)) # 3.54M params
self.mlp_down_B_bank = nn.Parameter(torch.empty(18, 512, 128))  # 1.18M params
```

---

## Training Dynamics

### Step Time Analysis

Low-rank MLP forward pass replaces one large matmul with two smaller ones:

```
Full-rank:     (B*T, 512) @ (512, 1536) = B*T*512*1536 FLOPs
Low-rank:      (B*T, 512) @ (512, 128) + (B*T, 128) @ (128, 1536)
             = B*T*128*(512 + 1536) = B*T*128*2048 FLOPs
Ratio: 128*2048 / (512*1536) = 0.334x
```

MLP is ~40% of total forward compute. Net per-step speedup: ~13%.
18 layers vs 11 layers: 64% more layers, so ~38% more forward compute per layer.
Net: 0.334 * 1.64 / 1.0 = 0.548x MLP time. Total step time: ~80-85ms vs 86ms (roughly neutral).

BUT with 18 layers, the attention compute also grows by 64%:
- Attention is ~50% of forward compute
- 1.64x attention + 0.55x MLP = 1.64*0.5 + 0.55*0.4 + 0.1 = 0.82 + 0.22 + 0.1 = 1.14x
- Step time: ~98ms -> ~6120 steps in 600s (vs 6920 for SOTA)

This is ~12% fewer steps. The depth gain must compensate.

### Expected BPB Derivation

**Gains:**
- Additional 7 layers at diminishing returns (~0.002 BPB/layer for layers 12-18):
  - 7 * 0.002 = 0.014 BPB improvement
  - But with diminishing rank quality, discount by 40%: **-0.0084 BPB**

**Costs:**
- Rank-128 quality loss vs full-rank per layer: ~+0.001-0.002 BPB per layer
  - Over 18 layers: +0.018-0.036 BPB (if measured per layer vs full-rank)
  - BUT: the model adapts to the constraint, so the actual penalty is much smaller
  - Estimated net rank penalty: **+0.003-0.005 BPB**
- Fewer training steps (12% fewer): **+0.001 BPB**

**Net: -0.0084 + 0.004 + 0.001 = -0.003 to -0.010 BPB** (range reflects uncertainty about rank quality)

The conservative estimate is -0.004 BPB. The optimistic estimate (if rank-128 learns well) is -0.010 BPB.

---

## Implementation Plan

### Files to Modify: `train_gpt.py`

1. **New class `LowRankLinear`** (~15 lines):
```python
class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)

    def forward(self, x):
        return x @ self.A @ self.B
```

2. **Modify MLP class** (~10 lines changed):
   - Replace `CastedLinear(dim, hidden)` with `LowRankLinear(dim, hidden, rank)`
   - Same for projection

3. **Modify GPT class** (~20 lines changed):
   - `NUM_LAYERS = 18`
   - Update U-Net skip connections: `num_encoder_layers = 9, num_decoder_layers = 9`
   - Update parameter banks to use factor shapes

4. **Modify Parallel Muon optimizer** (~10 lines changed):
   - Split MLP banks into A/B factor banks
   - Both are standard 2D -> Muon processes normally

5. **Modify quantization** (~5 lines changed):
   - For GPTQ: materialize W = A @ B, quantize as single int6 matrix (preserves GPTQ calibration quality)
   - For export: store the quantized materialized W (not the factors)

6. **Add env vars**: `MLP_RANK=128`, `NUM_LAYERS=18`

**Total: ~60-80 lines modified/added**

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Rank 128 is too aggressive (quality loss > depth gain) | MEDIUM | Approach fails | Try r=192 with 15 layers (safer, still +4 layers) |
| Thin matmuls are GPU-inefficient on H100 | MEDIUM | Step time slower than predicted | Batch factors across layers (contiguous memory) |
| Muon convergence changes with thin factors | LOW | Slower training | Use Newton-Schulz 7 steps for thin factors |
| GPTQ on materialized product loses calibration | LOW | Worse quantization | Quantize factors separately if product quant is poor |
| 18-layer U-Net skip connections overfit | LOW | Training instability | Use skip connections only for outer 6 layers |

---

## What Makes It NOVEL

**Not done by anyone on the leaderboard:**
- All 20 submissions use full-rank MLP weights
- The ternary submission (#10) used 73.7M params with ternary quantization but still full-rank MLPs
- No one has tried training with factored weights from scratch
- The depth progression stopped at 11 layers because full-rank MLPs hit the 16MB ceiling

**Distinct from post-hoc SVD/pruning:**
- SVD truncation discards learned information; low-rank training learns within the constraint
- The model's internal representations co-adapt with the rank constraint
- This is a fundamentally different loss landscape than training full-rank + truncating

---

## References

- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.
- Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." ACL 2021.
- Zhao et al. (2024). "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection." ICML 2024.
- Kamalakara et al. (2022). "Exploring Low Rank Training of Deep Neural Networks." arXiv:2209.13569.
- Parameter Golf depth progression: done.md entries #6 (10L), #1 (11L).
