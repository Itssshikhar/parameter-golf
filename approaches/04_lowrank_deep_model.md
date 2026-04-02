# Approach 4: Low-Rank MLP Training + Deeper Model

**Expected BPB improvement: -0.005 to -0.015**
**Artifact cost: Net zero (savings fund more layers)**
**Training cost: None (architecture change)**
**Risk: MEDIUM-HIGH**

---

## Problem Statement

The 16MB artifact limit caps the model at 11 layers (512d, 3× MLP, int6+LZMA). MLP weights consume 67% of all parameters (~17.3M of 26M total). If MLPs could be compressed in-training (not just post-training), the freed budget would fund additional layers. Each additional layer improves BPB by ~0.003-0.005 (from the 9→10→11 progression data).

## Core Idea

Replace full-rank MLP weight matrices (512×1536 and 1536×512) with trained low-rank factors (512×k and k×1536). The model learns to use the constrained representation from scratch, eliminating the truncation error of post-training SVD.

## Mathematical Formulation

### Standard MLP
```
h = ReLU²(W_fc · x)         W_fc: 512×1536  (786,432 params)
y = W_proj · h               W_proj: 1536×512 (786,432 params)
Total per layer: 1,572,864 params
Total 11 layers: 17,301,504 params
At int6+LZMA: ~12.0 MB
```

### Low-Rank MLP
```
h = ReLU²((B_fc · A_fc) · x)   A_fc: 512×k, B_fc: k×1536
y = (B_proj · A_proj) · h       A_proj: 1536×k, B_proj: k×512
Total per layer: 2 × k × (512 + 1536) = 4096k params
```

### Parameter Budget at Various Ranks

| Rank k | Params/layer | Total (11L) | At int6+LZMA | Savings vs full | Extra layers possible |
|--------|-------------|-------------|-------------|-----------------|----------------------|
| 64 | 262,144 | 2,883,584 | ~2.0 MB | ~10.0 MB | +13 layers |
| 96 | 393,216 | 4,325,376 | ~3.0 MB | ~9.0 MB | +12 layers |
| 128 | 524,288 | 5,767,168 | ~4.0 MB | ~8.0 MB | +10 layers |
| 192 | 786,432 | 8,650,752 | ~6.0 MB | ~6.0 MB | +8 layers |
| 256 | 1,048,576 | 11,534,336 | ~8.0 MB | ~4.0 MB | +5 layers |

### Proposed Configuration

**k=128, 18 layers** (7 additional layers):
```
Attention params (unchanged): 11 layers × 787K = 8.66M → 18 layers × 787K = 14.17M
Low-rank MLP params: 18 layers × 524K = 9.44M
Control params (scales, mix, skip, etc.): ~0.5M
Embeddings (1024×512, tied): 0.52M
BigramHash (3072×112): 0.40M

Total: ~25.0M params
At int6+LZMA: ~15.5-16.0 MB ✓
```

## Quality vs Rank Tradeoff

### Spectral Energy Analysis

A trained weight matrix W has singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_min(m,n). The fraction of spectral energy captured by rank-k approximation:

```
E(k) = (Σ_{i=1}^k σ_i²) / (Σ_{i=1}^{min(m,n)} σ_i²)
```

For the current SOTA MLP weights (trained with Muon + WD=0.04):
- Muon's Newton-Schulz orthogonalization flattens the singular value spectrum
- Weight decay (0.04) shrinks small singular values preferentially
- **Estimated spectral decay exponent α ≈ 0.7-1.0** (moderate to steep)

At α=0.8, rank k=128 captures ~92% of spectral energy for a 512×1536 matrix. The remaining 8% represents fine-grained weight patterns that the model would learn differently if constrained to rank 128.

### Key Insight: Training Low-Rank Is Better Than Truncating

Post-training SVD throws away the bottom singular values. Low-rank training forces the model to learn a different representation that maximally uses the available rank. The model adapts its feature learning to the constraint — it won't try to learn patterns that require rank > k. This is typically much better than truncation.

**Evidence from the literature:**
- LoRA (Hu et al., 2021): rank-8 LoRA matches full fine-tuning on many tasks
- Intrinsic dimensionality (Aghajanyan et al., 2021): most of the optimization happens in a low-dimensional subspace
- Low-rank pre-training (Zhao et al., 2024): achieves comparable quality with 30-50% of the parameters

## Implementation

### LowRankLinear Module

```python
class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.empty(rank, out_dim))
        # Orthogonal init for both factors
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)
        # Scale by 1/sqrt(2*num_layers) for projection layers
        self._zero_init = False

    def forward(self, x):
        return F.linear(F.linear(x, self.A.T), self.B.T)
```

### MLP with Low-Rank

```python
class LowRankMLP(nn.Module):
    def __init__(self, dim, mlp_mult, rank):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = LowRankLinear(dim, hidden, rank)    # 512→k→1536
        self.proj = LowRankLinear(hidden, dim, rank)   # 1536→k→512
        self.proj._zero_init = True

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
```

### Muon Compatibility

Muon applies Newton-Schulz orthogonalization to 2D weight matrices. The low-rank factors A (512×128) and B (128×1536) are standard 2D matrices — fully compatible.

The existing Muon scale correction `max(1, rows/cols)^0.5` handles rectangular matrices correctly:
- A (512×128): scale = sqrt(512/128) = 2.0
- B (128×1536): scale = 1.0 (128 < 1536)

### Parameter Banking Integration

The SOTA uses parameter banking (3D tensors: layers × rows × cols). For low-rank:
```python
# Instead of:
self.mlp_up_bank = nn.Parameter(torch.empty(11, 1536, 512))

# Use:
self.mlp_up_A_bank = nn.Parameter(torch.empty(18, 512, 128))   # 18 layers!
self.mlp_up_B_bank = nn.Parameter(torch.empty(18, 128, 1536))
```

The Parallel Muon optimizer processes these banks identically to the current full-rank banks.

## Training Dynamics

### Concern: Low-Rank Gradient Expressiveness

With full-rank weights, the gradient ∇_W L is an arbitrary m×n matrix. With factored weights A·B, the effective gradient is:

```
∇_A L = (∇_{AB} L) · B^T
∇_B L = A^T · (∇_{AB} L)
```

These lie in a restricted subspace — the gradient cannot escape the rank-k manifold. This is fine for convergence (the model learns the best rank-k representation) but may slow learning compared to full-rank.

### Mitigation: Larger Learning Rate

Low-rank factors benefit from slightly higher learning rates because the gradient signal per parameter is stronger (each parameter influences more output dimensions). Recommend:
- `matrix_lr = 0.030` (vs 0.025 for full-rank)
- `muon_momentum = 0.99` (unchanged)

### Step Time Analysis

Low-rank MLP forward pass: two smaller matmuls instead of one large:
```
Full: (B*T, 512) × (512, 1536) = B*T*512*1536 FLOPs
Low-rank: (B*T, 512) × (512, 128) + (B*T, 128) × (128, 1536) = B*T*128*(512+1536) FLOPs
Ratio: 128*2048 / (512*1536) = 262144 / 786432 = 0.33×
```

MLP forward is ~33% the cost. But attention is unchanged, and MLP is ~40% of total forward. Net step time reduction: ~13%. At 86ms → ~75ms → ~8000 steps in 600s (vs 6920).

### Combined Effect: More Layers + Faster Steps
- 18 layers at 75ms/step → 8000 steps in 600s → 6.3B tokens
- vs 11 layers at 86ms/step → 6920 steps → 5.4B tokens
- **17% more tokens AND 64% more layers**

## Expected BPB Derivation

From the 9→10→11 layer progression:
- 9→10: -0.0032 BPB per layer
- 10→11: -0.0025 BPP per layer (diminishing returns)
- Average: ~-0.003 per layer

At rank k=128 with quality loss:
- Spectral truncation penalty: ~+0.005 to +0.010 BPP (partially recovered by training low-rank)
- Additional 7 layers: ~-0.015 to -0.021 BPB
- Faster steps (more training): ~-0.002 BPB

**Net: -0.005 to -0.015 BPB** (wide range reflecting uncertainty about rank-128 quality)

## Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Rank 128 too aggressive (>10% quality loss) | MEDIUM | +0.015 BPB regression from rank | Try k=192 (8 extra layers instead of 7) |
| Muon convergence slower with low-rank | LOW | Fewer effective training steps | Increase LR, extend warmup |
| GPTQ behaves differently on thin factors | LOW | Worse quantization | GPTQ on the product A·B (materialize then quantize) |
| Small matmuls inefficient on H100 | MEDIUM | Step time not as fast as predicted | Batch matmuls across layers |

## Alternatives Considered

### Kronecker Factorization
W ≈ A ⊗ B where A is (48×32) and B is (32×16). Even more aggressive compression (438× per matrix) but:
- Small Kronecker factors are GPU-inefficient
- Training convergence is poorly understood for LMs
- Not recommended as primary approach

### Shared Spectral Basis
W_l = C_l × B where B is shared across layers. 3.88× compression but:
- Forces all layers into the same subspace
- Too rigid for the heterogeneous roles of different layers

## References

- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
- Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning."
- Zhao et al. (2024). "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection."
- Parameter Golf depth progression: 9L→10L→11L improvements documented in done.md.
