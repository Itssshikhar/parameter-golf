# Approach 3: Spectral Weight Compression (SVD + Quantized Factors)

**Type: ARCHITECTURE + COMPRESSION change (affects training via QAT)**
**Expected BPB improvement: -0.002 to -0.006**
**Artifact cost: Saves 1-3 MB vs int6 scalar quantization**
**Training cost: ~5% overhead for SVD-aware QAT**
**Risk: MEDIUM**

---

## Problem Statement

Current SOTA uses int6 scalar quantization: each weight is stored as a 6-bit integer plus a per-row scale factor. This achieves ~4.2x compression over fp32, but the quantization error is distributed uniformly across all singular directions of the weight matrix. This is suboptimal because:

1. **Top singular directions carry more information.** The top singular values of trained weights encode the most important learned features. Quantization noise in these directions hurts BPB disproportionately.
2. **Bottom singular directions are noise-dominated.** After training with weight decay 0.04, small singular values are pushed toward zero. Storing them with the same bit precision as large values wastes bits.

**Solution: Replace int6 scalar quantization with SVD factorization + asymmetric quantization of the factors.** The top singular directions get higher precision; the bottom ones get fewer bits or are discarded entirely. The freed bytes fund either more layers or larger BigramHash.

This is a compression/architecture change that also requires SVD-aware QAT during training.

---

## Mathematical Formulation

### Current: Int6 Scalar Quantization

For a weight matrix W of shape (m, n):
```
For each row i:
  scale_i = percentile(|W[i, :]|, 99.99984) / 31
  Q[i, :] = clamp(round(W[i, :] / scale_i), -31, 31)

Storage: m*n * 6 bits + m * 16 bits (scales)
Reconstruction: W_hat[i, :] = Q[i, :] * scale_i
MSE = sum_ij (W[i,j] - W_hat[i,j])^2
```

The quantization error has equal variance across all directions. The MSE contribution from the k-th singular direction is:
```
MSE_k = sigma_k^2 * (quantization_noise_variance)
```

This is the same for every k. But the impact on the loss function is NOT uniform -- perturbations along the top singular directions have larger effect on the output.

### Proposed: SVD + Asymmetric Quantized Factors

**Step 1: Compute truncated SVD**
```
W = U * S * V^T     where U: (m, r), S: (r, r), V: (n, r)
W_approx = U * S * V^T    (truncated to rank r)
```

**Step 2: Absorb S into factors**
```
A = U * sqrt(S)     shape: (m, r)
B = sqrt(S) * V^T   shape: (r, n)
W_approx = A * B
```

**Step 3: Quantize A and B asymmetrically**
```
A_q = quantize(A, bits_A)    # Top factors get more bits
B_q = quantize(B, bits_B)    # Can use different bit width
W_hat = A_q * B_q
```

### Storage Analysis

For a weight matrix W: (512, 1536) = 786,432 elements

**Int6 scalar quantization:**
```
Storage = 786,432 * 6/8 + 512 * 2 = 589,824 + 1,024 = 590,848 bytes
```

**SVD rank-256 + int8 factors:**
```
A: (512, 256) at int8 = 131,072 bytes + 512 scales * 2 = 132,096 bytes
B: (256, 1536) at int8 = 393,216 bytes + 256 scales * 2 = 393,728 bytes
Total = 525,824 bytes
Savings = 590,848 - 525,824 = 65,024 bytes (11% savings)
```

**SVD rank-256 + int6 A + int8 B (asymmetric):**
```
A: (512, 256) at int6 = 98,304 bytes + 512 * 2 = 99,328 bytes
B: (256, 1536) at int8 = 393,216 bytes + 256 * 2 = 393,728 bytes
Total = 493,056 bytes
Savings = 590,848 - 493,056 = 97,792 bytes (16.5% savings)
```

**SVD rank-192 + int8 factors:**
```
A: (512, 192) at int8 = 98,304 + 1,024 = 99,328 bytes
B: (192, 1536) at int8 = 294,912 + 384 = 295,296 bytes
Total = 394,624 bytes
Savings = 590,848 - 394,624 = 196,224 bytes (33% savings)
```

### Per-Layer and Total Savings

Each layer has 2 MLP matrices + 4 attention matrices. The MLP matrices (512x1536, 1536x512) dominate.

For 11 layers with SVD rank-256 + int6/int8 asymmetric:
```
MLP savings per layer: 2 * 97,792 = 195,584 bytes
Attention matrices are smaller (512x512, 512x256) -- less SVD benefit
Estimated attention savings per layer: ~30,000 bytes
Total per layer: ~225,000 bytes = ~220 KB
Total 11 layers: ~2.4 MB savings
```

**With 2.4 MB freed, we can fit 2 additional layers** (each layer at int6 = ~1.1 MB) --> **13 layers total**.

### Quality of SVD Rank-256 Approximation

For a (512, 1536) matrix trained with Muon + WD=0.04, the singular value distribution follows an approximately power-law decay:
```
sigma_k ~ k^{-alpha}    where alpha ~ 0.7-1.0 (Muon flattens the spectrum)
```

The fraction of Frobenius norm captured by rank-256 out of min(512, 1536) = 512:
```
E(256) = sum_{k=1}^{256} k^{-2*alpha} / sum_{k=1}^{512} k^{-2*alpha}

For alpha = 0.8:
Numerator   = sum_{k=1}^{256} k^{-1.6} ~ H_{256}^{(1.6)} ~ 6.32
Denominator = sum_{k=1}^{512} k^{-1.6} ~ H_{512}^{(1.6)} ~ 6.89
E(256) ~ 91.7%

For alpha = 1.0:
E(256) ~ 95.8%
```

Discarding 4-8% of spectral energy while gaining int8 precision on the retained directions is a favorable trade when the discarded directions are noise-dominated.

### SVD-Aware QAT During Training

To minimize the mismatch between training and quantized inference, we add SVD-aware fake quantization during the late QAT phase:

```python
def svd_quantize_ste(W, rank, bits_A=6, bits_B=8):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    A = U_r * S_r.sqrt().unsqueeze(0)      # (m, r)
    B = S_r.sqrt().unsqueeze(1) * Vh_r     # (r, n)
    A_q = fake_quantize(A, bits_A)
    B_q = fake_quantize(B, bits_B)
    W_hat = A_q @ B_q
    # STE: use W_hat in forward, but gradients flow to W
    return W + (W_hat - W).detach()
```

This teaches the network to be robust to SVD truncation + factor quantization.

---

## How It Builds on the SOTA Stack

| Component | Change? | Details |
|-----------|---------|---------|
| Training (steps 0 to QAT start) | No change | Standard Muon+Adam |
| Late QAT | **MODIFIED** | SVD-aware fake quantization replaces int6 STE |
| Post-training quantization | **REPLACED** | SVD + factor quantization instead of GPTQ int6 |
| LZMA compression | **ADAPTED** | Compress SVD factors (may compress differently) |
| Architecture | **EXPANDED** | 2 more layers funded by byte savings |
| Everything else | No change | XSA, BigramHash, Parallel Muon, EMA, etc. |

---

## Expected BPB Improvement

**Gains:**
- 2 additional layers (11 -> 13): ~2 * 0.002 = **-0.004 BPB**
  (Using diminishing returns rate from layer 10-11 data)

**Costs:**
- SVD rank truncation (discarding 4-8% spectral energy): **+0.001-0.002 BPB**
  (Mitigated by SVD-aware QAT which adapts the model to the truncation)
- Factor quantization error (int6/int8 on factors vs int6 on full matrix): **~0.000 BPB**
  (int8 factors have lower per-element error than int6 on full matrix)

**Net: -0.004 + 0.0015 = -0.002 to -0.006 BPB**

The variance comes from uncertainty about:
1. Actual spectral decay rate of Muon-trained weights (alpha)
2. LZMA compressibility of SVD factors vs scalar quantized weights
3. Whether the freed bytes are exactly enough for 2 layers

---

## Implementation Plan

### Files to Modify: `train_gpt.py`

1. **Add `svd_quantize_ste()` function** (~20 lines):
   - Compute truncated SVD
   - Factor absorption (sqrt(S) into both A and B)
   - Fake quantize A and B with specified bit widths
   - Return STE-compatible output

2. **Modify `CastedLinear.forward()` for SVD-aware QAT** (~10 lines):
   - When `_qat_enabled and training`: call `svd_quantize_ste()` instead of int6 fake quantize
   - Same STE gradient flow

3. **Add `quantize_svd_factors()` post-training function** (~40 lines):
   - For each weight matrix: compute SVD, truncate to rank r
   - Absorb singular values into factors
   - Quantize each factor with specified bit width
   - Store factors + scales in state dict

4. **Add `dequantize_svd_factors()` for inference** (~20 lines):
   - Reconstruct W = dequant(A_q) @ dequant(B_q)
   - Can be fused into the forward pass (two matmuls instead of one, or materialized once)

5. **Modify model architecture** (~10 lines):
   - Increase NUM_LAYERS from 11 to 13
   - Update U-Net skip connections (6 encoder + 7 decoder)

6. **Add env vars** (~5 lines):
   - `SVD_RANK=256`
   - `SVD_BITS_A=6`
   - `SVD_BITS_B=8`
   - `NUM_LAYERS=13` (enabled by byte savings)

**Total: ~100-120 lines modified/added**

### Testing Strategy

1. **Measure spectral decay** of current SOTA weights:
   ```python
   for name, W in model.state_dict().items():
       if W.dim() == 2 and W.shape[0] >= 256:
           S = torch.linalg.svdvals(W.float())
           energy_256 = (S[:256]**2).sum() / (S**2).sum()
           print(f"{name}: E(256) = {energy_256:.4f}")
   ```

2. **Measure compression ratio** of SVD factors vs scalar int6:
   - Export both formats, LZMA compress, compare sizes

3. **A/B test**: Train 11L with SVD-QAT vs standard QAT. If SVD-QAT matches or beats standard int6 quantization quality, the approach validates.

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| SVD computation too slow for QAT (every step) | MEDIUM | Training overhead > 10% | Compute SVD every 10 steps, cache the basis |
| Muon-trained weights have flat spectrum (alpha < 0.5) | LOW | SVD truncation is costly | Use rank-384 (less truncation, less savings) |
| LZMA compresses SVD factors poorly | MEDIUM | Byte savings smaller than estimated | SVD factors are smoother than scalar quantized; likely compress better |
| Two-matmul inference is slower than one | LOW | Eval time increases | Materialize W = A @ B once at load time |
| QAT gradient through SVD is noisy | MEDIUM | Training instability late | Use small SVD QAT LR, or only apply SVD in last 200 steps |
| torch.linalg.svd is numerically unstable in bf16 | LOW | NaN during QAT | Cast to fp32 for SVD, back to bf16 for matmuls |

---

## What Makes It NOVEL

**Not done by anyone on the leaderboard:**
- All submissions use scalar per-row quantization (int5, int6, int8) or ternary
- No submission has used SVD-based weight compression
- No submission has asymmetric bit allocation across singular directions
- The GPTQ variants (GPTQ-lite clip search) still operate on scalar quantization

**Distinct from post-hoc SVD pruning:**
- SVD-aware QAT during training means the model adapts to the SVD truncation
- The factors are quantized (not stored in fp16/fp32), combining SVD compression with quantization
- This is a principled information-theoretic approach: allocate bits proportionally to importance

**Novel combination:**
- SVD for rank reduction + quantization for bit reduction = multiplicative compression
- Neither SVD alone nor quantization alone achieves this compression ratio

---

## References

- Hsu et al. (2022). "Language Model Compression with Weighted Low-Rank Factorization." ICLR 2022.
- Frantar et al. (2023). "GPTQ: Accurate Post-Training Quantization." ICLR 2023.
- Chee et al. (2024). "QuIP#: Even Better LLM Quantization with Hadamard Incoherence." ICML 2024.
- Kovaleva et al. (2024). "ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs." arXiv:2312.05821.
- Zhang et al. (2024). "LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning." ACL Findings 2024.
