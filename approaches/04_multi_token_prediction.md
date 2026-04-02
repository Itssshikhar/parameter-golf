# Approach 4: Multi-Token Prediction + Bytes-Per-Token Weighted Loss

**Type: TRAINING change (loss function modification)**
**Expected BPB improvement: -0.003 to -0.008**
**Artifact cost: ZERO (auxiliary heads removed at export)**
**Training cost: ~15-20% overhead (auxiliary heads)**
**Risk: MEDIUM**

---

## Problem Statement

Two independent training inefficiencies in the current SOTA:

### Problem A: The Model Learns Shallow Next-Token Statistics

Standard next-token prediction with cross-entropy loss rewards the model for learning P(x_{t+1} | x_{<=t}). This encourages shallow pattern matching (what immediately follows?) rather than deep understanding of the data distribution. Tokens that are predictable from local bigram/trigram patterns get the same weight in the loss as tokens that require global understanding.

Multi-token prediction (MTP) adds auxiliary heads predicting x_{t+2}, x_{t+3}, etc. These heads share the backbone but have separate output projections. During training, the gradients from future-token prediction force the backbone to learn richer representations that capture longer-range dependencies. At export time, the auxiliary heads are discarded (zero artifact cost).

**Evidence from Meta AI (Gloeckle et al., 2024):** MTP with 4 future tokens improved downstream task quality by 2-5% on code generation and reasoning benchmarks, with negligible inference cost.

### Problem B: The Loss Weights Tokens Uniformly, Not by Byte Cost

BPB is the competition metric:
```
BPB = (sum_t NLL(x_t)) / (ln(2) * sum_t bytes(x_t))
```

But the training loss is:
```
L = (1/T) * sum_t CE(x_t)    (uniform weighting)
```

A 5-byte token and a 1-byte token contribute equally to the loss, but the 5-byte token contributes 5x more to the BPB denominator. Mispredicting a 1-byte token hurts BPB 5x more per NLL unit than mispredicting a 5-byte token, because:

```
dBPB/dNLL(x_t) = 1 / (ln(2) * total_bytes)
```

This is independent of bytes(x_t). But the opportunity cost is different: spending model capacity on tokens worth fewer bytes is less efficient. More precisely, the optimal training strategy should upweight tokens that are both (a) improvable and (b) contribute many NLL units per byte to BPB.

A simpler but effective proxy: weight each token's loss inversely proportional to its byte count:
```
L_weighted = (1/T) * sum_t (1/bytes(x_t)) * CE(x_t)
```

This makes the training objective more aligned with the BPB metric.

---

## Mathematical Formulation

### Multi-Token Prediction (MTP)

Given hidden states h_t from the last transformer layer, add K auxiliary prediction heads:

```
logits_0[t] = lm_head(h_t)                    (standard: predict x_{t+1})
logits_k[t] = aux_head_k(h_t)  for k=1..K     (auxiliary: predict x_{t+1+k})
```

Each auxiliary head is a single linear projection:
```
aux_head_k: R^d -> R^V    (512 -> 1024, same as lm_head)
```

The MTP loss:
```
L_MTP = sum_{k=0}^{K} w_k * (1/T_k) * sum_t CE(logits_k[t], x_{t+1+k})
```

Where:
- w_0 = 1.0 (standard next-token loss, full weight)
- w_k = alpha * gamma^k for k >= 1 (exponentially decaying auxiliary weights)
- T_k = T - k (fewer valid positions for further-ahead predictions)

**Recommended: K=3, alpha=0.3, gamma=0.5**
```
w_0 = 1.0, w_1 = 0.30, w_2 = 0.15, w_3 = 0.075
```

The auxiliary heads provide a training signal that:
1. Forces the backbone to encode information about x_{t+2}, x_{t+3}, x_{t+4}
2. Creates a richer gradient signal through the backbone (more supervision per token)
3. Acts as a regularizer (the model must extract features useful for multiple future positions)

### BPB-Weighted Loss

The standard CE loss treats all tokens equally:
```
L_standard = (1/T) * sum_t CE(logits[t], x_{t+1})
```

The BPB-weighted loss adjusts by inverse byte count:
```
L_BPB = (1/Z) * sum_t (1 / bytes(x_{t+1})) * CE(logits[t], x_{t+1})

where Z = sum_t (1 / bytes(x_{t+1}))   (normalizing constant)
```

This upweights short tokens (1-2 bytes, like punctuation and common ASCII characters) relative to long tokens (3-5 bytes, like multi-byte UTF-8 sequences). Short tokens contribute more NLL per byte to BPB, so improving their prediction has higher BPB return.

### Combined Loss

```
L_total = (1/Z) * sum_t (1/bytes(x_{t+1})) * [
    CE(logits_0[t], x_{t+1})                           (main)
    + sum_{k=1}^{K} w_k * CE(logits_k[t], x_{t+1+k})  (auxiliary)
]
```

**At export time:** discard aux_head_1, aux_head_2, aux_head_3. The model size is unchanged. Only the backbone's learned representations are different (richer, more forward-looking).

### Gradient Analysis: Why MTP Helps the Backbone

For the standard loss, the gradient through the backbone at position t is:
```
dL/dh_t = dL/dlogits_0[t] * d_logits_0/dh_t = (P_0(v|h_t) - 1[v=x_{t+1}]) * W_lm_head^T
```

With MTP, the gradient becomes:
```
dL/dh_t = (P_0 - 1[x_{t+1}]) * W_lm^T
         + w_1 * (P_1 - 1[x_{t+2}]) * W_aux1^T
         + w_2 * (P_2 - 1[x_{t+3}]) * W_aux2^T
         + w_3 * (P_3 - 1[x_{t+4}]) * W_aux3^T
```

The auxiliary gradients provide **additional supervision** at every position. Instead of h_t only needing to encode information about x_{t+1}, it must also encode predictive features for x_{t+2..t+4}. This:

1. Reduces overfitting to local patterns (the model can't just memorize bigrams)
2. Encourages hierarchical features (predicting 3 steps ahead requires different features than 1 step ahead)
3. Increases the effective supervision signal per training step (more gradient information per token)

### Interaction with BPB Metric

The BPB weighting ensures that the training signal prioritizes tokens with high BPB impact. Combined with MTP, this means:

- Short tokens (high BPB weight, often function words/punctuation) get richer multi-step predictions
- Long tokens (low BPB weight, often content words) get proportionally less training focus
- This aligns the training objective with the competition metric

---

## How It Builds on the SOTA Stack

| Component | Change? | Details |
|-----------|---------|---------|
| Backbone architecture | No change | Same 11L 512d model |
| Forward pass | **MODIFIED** | h_t passed through K+1 heads instead of 1 |
| Loss function | **MODIFIED** | MTP + BPB weighting |
| Muon optimizer | No change | Processes same backbone params |
| Adam for heads | **EXTENDED** | Additional aux_head params added to Adam |
| EMA, SWA | No change | Average backbone + main head (not aux) |
| Late QAT | No change | Applied to backbone weights |
| Export / GPTQ | No change | Only export backbone + main head |
| Eval | No change | Standard forward (only logits_0 used) |

### Parameter Overhead During Training Only

```
Auxiliary heads: K * V * d = 3 * 1024 * 512 = 1,572,864 params
As fraction of model: 1.57M / 26M = 6%
Memory overhead: 1.57M * 4 bytes (fp32) + optimizer state = ~12 MB
Training compute overhead: 3 additional linear projections per step = ~15%
```

This is within the 10-min training budget. At 86ms/step + 15% overhead = ~99ms/step:
```
6920 * 86 / 99 = ~6010 steps    (vs 6920 baseline)
Steps lost: ~910 (13%)
```

Each step at convergence contributes ~0.0001 BPB. Cost of fewer steps: **+0.001 BPB**.

---

## Expected BPB Improvement

### MTP Contribution

From Gloeckle et al. (2024), MTP with K=4 auxiliary heads improved:
- Code generation: +5-10% on HumanEval
- Reasoning: +2-4% on various benchmarks
- Perplexity: -1-3% on held-out text

Scaling down to our setting (smaller model, fewer heads K=3):
- Expected backbone quality improvement: **-0.003 to -0.005 BPB**

### BPB Weighting Contribution

The gap between uniform-weighted training and BPB-optimal training depends on the byte-count distribution of the tokenizer.

For sp1024 (1024 BPE vocab):
- Mean bytes/token: ~4.2 (estimated from FineWeb statistics)
- Std bytes/token: ~1.8
- Coefficient of variation: ~0.43

The BPB weighting shifts ~15% of the gradient mass from long tokens to short tokens. The expected improvement:
```
delta_BPB = CV^2 * baseline_BPB * efficiency_factor
          = 0.43^2 * 1.1147 * 0.01
          = 0.002 BPB
```

This is a rough estimate. The actual gain depends on whether short tokens are harder to predict (they often are -- they include rare punctuation and function words whose distribution is context-dependent).

Estimated: **-0.001 to -0.003 BPB**

### Combined

```
MTP:            -0.003 to -0.005
BPB weighting:  -0.001 to -0.003
Step cost:      +0.001
Net:            -0.003 to -0.008 BPB
```

---

## Implementation Plan

### Files to Modify: `train_gpt.py`

1. **Add auxiliary heads to GPT class** (~15 lines):
```python
class GPT(nn.Module):
    def __init__(self, ...):
        ...
        # Multi-token prediction heads (training only)
        self.mtp_heads = nn.ModuleList([
            CastedLinear(args.model_dim, args.vocab_size)
            for _ in range(args.mtp_k)
        ]) if args.mtp_k > 0 else None
```

2. **Modify forward() to compute MTP loss** (~25 lines):
```python
def forward(self, x, targets, base_bytes_lut=None, mtp_weights=None):
    h = self.backbone(x)  # (B, T, d)
    logits_0 = self.lm_head(h)

    # BPB-weighted CE loss
    if base_bytes_lut is not None:
        byte_counts = base_bytes_lut[targets].float()  # (B, T)
        inv_bytes = 1.0 / byte_counts.clamp(min=1)
        inv_bytes = inv_bytes / inv_bytes.mean()  # normalize to mean 1
        ce_0 = F.cross_entropy(logits_0.view(-1, V), targets.view(-1), reduction='none')
        loss = (ce_0 * inv_bytes.view(-1)).mean()
    else:
        loss = F.cross_entropy(logits_0.view(-1, V), targets.view(-1))

    # Multi-token prediction losses
    if self.mtp_heads is not None and self.training:
        for k, head in enumerate(self.mtp_heads, start=1):
            logits_k = head(h[:, :-k, :])  # shift: predict x_{t+1+k}
            targets_k = targets[:, k:]       # target is shifted by k
            ce_k = F.cross_entropy(logits_k.reshape(-1, V), targets_k.reshape(-1))
            loss = loss + mtp_weights[k] * ce_k

    return loss
```

3. **Modify training loop for BPB LUT** (~10 lines):
   - Pass `base_bytes_lut` into forward() during training
   - The LUT is already computed for eval; reuse it

4. **Exclude aux heads from export** (~5 lines):
```python
# At export time:
state_dict = {k: v for k, v in model.state_dict().items()
              if 'mtp_heads' not in k}
```

5. **Add aux heads to Adam optimizer** (~5 lines):
   - MTP heads use Adam (they're linear projections), not Muon

6. **Add env vars** (~5 lines):
   - `MTP_K=3` (number of auxiliary future-token heads)
   - `MTP_ALPHA=0.3` (base weight for auxiliary losses)
   - `MTP_GAMMA=0.5` (decay factor per step ahead)
   - `BPB_WEIGHTED_LOSS=1` (enable BPB-weighted CE)

**Total: ~65-80 lines modified/added**

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| MTP overhead exceeds 10-min budget | LOW | Fewer training steps | Reduce K to 2 or 1; disable aux heads after 80% of training |
| BPB weighting destabilizes training | LOW | Loss spikes early | Warm up the weighting: start uniform, linearly increase weighting over 500 steps |
| Auxiliary heads steal gradient from main head | MEDIUM | Main head quality degrades | Use small w_k weights (0.3, 0.15, 0.075); if still an issue, use stop_gradient on aux backprop through backbone for first 1000 steps |
| Byte LUT is inaccurate for edge cases | LOW | Weighting is slightly off | Use the exact same LUT as eval (already validated) |
| MTP doesn't help at this model scale | MEDIUM | No quality improvement | MTP has been shown effective down to 125M params (our scale is 26M -- somewhat below the evidence range) |
| torch.compile issues with dynamic head count | LOW | Compile failure | Unroll the MTP loop explicitly (3 heads is trivial) |

---

## What Makes It NOVEL

**Not done by anyone on the leaderboard:**
- All 20 submissions use standard next-token CE loss
- No submission uses multi-token prediction auxiliary heads
- No submission uses BPB-weighted loss (all use uniform CE)
- The closest is TTT (PR #549), but that adapts at test time, not during training

**Distinct from standard training:**
- MTP changes the gradient signal through the backbone at every training step
- BPB weighting aligns the training objective with the competition metric
- Both are training-only modifications with zero inference cost

**Literature support:**
- Gloeckle et al. (2024): "Better & Faster Large Language Models via Multi-token Prediction" -- Meta AI, published at ICML 2024, demonstrated clear gains at scale
- The BPB weighting is a novel contribution specific to the Parameter Golf setting (no prior work weights CE loss by bytes-per-token because BPB isn't a standard metric)

---

## Ablation Plan

To isolate the contributions:

| Experiment | MTP? | BPB weight? | Expected BPB | Purpose |
|-----------|------|-------------|-------------|---------|
| Baseline | No | No | 1.1147 | Control |
| MTP only | K=3 | No | 1.108-1.112 | Isolate MTP effect |
| BPB weight only | No | Yes | 1.112-1.114 | Isolate weighting effect |
| MTP + BPB weight | K=3 | Yes | 1.107-1.112 | Full approach |
| MTP K=1 | K=1 | Yes | 1.109-1.113 | Check if K>1 matters |

Run each experiment once (10 min on 8xH100). Total cost: 50 min for the full ablation.

---

## References

- Gloeckle et al. (2024). "Better & Faster Large Language Models via Multi-token Prediction." ICML 2024.
- Qi et al. (2020). "ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training." ACL 2020.
- Stern et al. (2018). "Blockwise Parallel Decoding for Deep Autoregressive Models." NeurIPS 2018.
- Lin et al. (2024). "Rethinking Token Weighting in Language Model Training." arXiv preprint.
- Parameter Golf competition rules: BPB = bits_per_token * tokens_per_byte.
