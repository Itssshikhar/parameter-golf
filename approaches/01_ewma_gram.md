# Approach 1: EWMA-gram — Online Frequency Mixing at Eval Time

**Expected BPB improvement: -0.003 to -0.012**
**Artifact cost: ZERO bytes**
**Training cost: ZERO**
**Eval cost: ~1-2 seconds**
**Risk: LOW**

---

## Problem Statement

The neural model generalizes well across documents but cannot memorize document-specific patterns encountered during evaluation (repeated names, technical terms, code identifiers, stylistic tics). N-gram caching could capture these, but 33+ PRs were disqualified for improper normalization (hash-based lookups that didn't produce valid probability distributions).

## Core Idea

Maintain exponentially-weighted moving average (EWMA) frequency counters during evaluation, updated ONLY from already-scored tokens. Mix the resulting distributions with the neural model's predictions using entropy-adaptive weights. The frequency counters produce properly normalized distributions by construction.

## Mathematical Formulation

### Counter Update (after scoring token x_t with context token c = x_{t-1})

**Unigram EWMA:**
```
count_uni[v] *= λ_uni           ∀v ∈ [0, V)
count_uni[x_t] += (1 - λ_uni)
P_uni(v) = count_uni[v] / Σ_w count_uni[w]
```

**Bigram EWMA:**
```
bucket = hash(c) % n_buckets
count_bi[bucket][v] *= λ_bi           ∀v ∈ [0, V)
count_bi[bucket][x_t] += (1 - λ_bi)
P_bi(v) = count_bi[bucket][v] / Σ_w count_bi[bucket][w]
```

### Normalization Proof
```
Σ_v P_bi(v) = Σ_v count_bi[bucket][v] / Σ_w count_bi[bucket][w] = 1  ✓
```
This is guaranteed for any non-negative count vector with positive sum. The EWMA update preserves non-negativity and positivity of the sum (initialized to uniform 1/V, all operations maintain this).

### Entropy-Adaptive Mixing
```
H_neural = -Σ_v P_neural(v) · log P_neural(v)     (model entropy at this position)
σ = sigmoid(w · H_neural + b)                       (learned gate, 2 scalar params)
α_bi = σ · α_bi_max                                 (high α when model uncertain)
α_uni = σ · α_uni_max                               (smaller contribution)
P_final(v) = (1 - α_bi - α_uni) · P_neural(v) + α_bi · P_bi(v) + α_uni · P_uni(v)
```

### Final Loss
```
NLL(x_t) = -log P_final(x_t)
BPB = (Σ_t NLL(x_t) / ln(2)) / total_bytes
```

### Normalization of P_final
```
Σ_v P_final(v) = (1-α_bi-α_uni)·1 + α_bi·1 + α_uni·1 = 1  ✓
```

## Hyperparameters

| Parameter | Description | Suggested Value | Tuning Range |
|-----------|-------------|-----------------|--------------|
| λ_uni | Unigram decay rate | 0.999 | [0.99, 0.9999] |
| λ_bi | Bigram decay rate | 0.995 | [0.99, 0.999] |
| α_bi_max | Max bigram mixing weight | 0.10 | [0.05, 0.20] |
| α_uni_max | Max unigram mixing weight | 0.03 | [0.01, 0.10] |
| w | Entropy gate weight | 2.0 | [0.5, 5.0] |
| b | Entropy gate bias | -3.0 | [-6.0, 0.0] |
| n_buckets | Bigram hash buckets | 1024 | [512, 4096] |

Total stored parameters: 6 scalars = 24 bytes (can be hardcoded or tuned on training set).

## Memory Requirements

- Unigram counters: V × 4 bytes = 1024 × 4 = **4 KB**
- Bigram counters: n_buckets × V × 4 bytes = 1024 × 1024 × 4 = **4 MB**
- Total runtime memory: **~4 MB** (trivial on 80GB H100)

## Legality Analysis

**Legal under established precedent.** The competition rules state: "you are only allowed to test-time train on validation set tokens you've already evaluated your model on, since those tokens have already been graded."

- Legal TTT (PR #549, accepted as SOTA at 1.1194 BPB) adapts MODEL WEIGHTS using SGD on scored tokens — far more aggressive than EWMA-gram
- EWMA-gram only accumulates summary statistics (frequency counts), never modifies model weights
- Counters are updated AFTER scoring, guaranteeing backward-looking-only property
- The score for token x_t is computed BEFORE the counter is updated with x_t

## Interaction with Other Approaches

- **Composable with all 4 other approaches** — EWMA-gram modifies only the eval-time probability combination, independent of model architecture, training, and quantization
- **Complementary to sliding window eval** — sliding window improves context for the neural model; EWMA-gram captures cross-window repetition patterns
- **Partially overlaps with TTT** — both adapt to the document at eval time, but via different mechanisms (weight adaptation vs. statistics accumulation). Can be used together.

## Expected BPB Derivation

From Grave et al. (2016) "Improving Neural Language Models with a Continuous Cache": neural cache models achieve 10-15% perplexity reduction on WikiText-103. The Parameter Golf setting differs:
- Smaller vocab (1024 vs 267K) → fewer hash collisions, better n-gram coverage
- Shorter documents (~1240 tokens avg) → less local repetition to exploit
- Stronger base model → less room for improvement

Conservative estimate: 3-5% of the gap between neural model (1.11 BPB) and theoretical limit (~0.9 BPB) = **0.003-0.010 BPB improvement**.

## Implementation Sketch

```python
def eval_val_sliding_ewma(args, model, rank, world_size, device, val_tokens,
                          base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                          stride=64, lambda_uni=0.999, lambda_bi=0.995,
                          alpha_bi_max=0.10, alpha_uni_max=0.03,
                          w_gate=2.0, b_gate=-3.0, n_buckets=1024):
    V = args.vocab_size
    seq_len = args.eval_seq_len or args.train_seq_len

    # Initialize EWMA counters (uniform prior)
    count_uni = torch.ones(V, device=device) / V
    count_bi = torch.ones(n_buckets, V, device=device) / V

    # Standard sliding window setup...
    model.eval()
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for batch in windows:
            x, y = batch  # [B, seq_len]

            # Neural model forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)  # Need logits, not loss
            P_neural = torch.softmax(logits.float(), dim=-1)  # [B, T, V]

            # For each scored position in this window:
            for pos in scored_positions:
                target = y[:, pos]
                context_tok = x[:, pos]

                # Bigram distribution
                bucket = (context_tok % n_buckets).long()
                P_bi = count_bi[bucket] / count_bi[bucket].sum(dim=-1, keepdim=True)

                # Unigram distribution
                P_uni = count_uni / count_uni.sum()

                # Entropy-adaptive mixing
                p_n = P_neural[:, pos]
                H = -(p_n * (p_n + 1e-10).log()).sum(dim=-1)
                sigma = torch.sigmoid(w_gate * H + b_gate)
                a_bi = sigma * alpha_bi_max
                a_uni = sigma * alpha_uni_max

                # Mixed distribution
                P_final = (1 - a_bi - a_uni).unsqueeze(-1) * p_n \
                        + a_bi.unsqueeze(-1) * P_bi \
                        + a_uni.unsqueeze(-1) * P_uni.unsqueeze(0)

                # Score
                nll = -P_final.gather(-1, target.unsqueeze(-1)).squeeze(-1).log()
                val_loss_sum += nll.sum()
                val_token_count += target.numel()

                # Update counters (AFTER scoring)
                count_uni *= lambda_uni
                count_uni.scatter_add_(0, target, torch.full_like(target, 1-lambda_uni, dtype=count_uni.dtype))
                count_bi[bucket] *= lambda_bi
                count_bi[bucket].scatter_add_(-1, target.unsqueeze(-1),
                    torch.full((target.shape[0], 1), 1-lambda_bi, device=device))

                # Byte counting (same as standard eval)
                # ...

    # All-reduce across ranks, compute BPB
    # ...
```

**Note**: The above is a conceptual sketch. The actual implementation must handle:
1. Batched processing across the sliding window
2. Per-rank counter state (each GPU processes contiguous temporal blocks)
3. Efficient scatter_add_ for GPU counter updates
4. Integration with the existing `eval_val_sliding` function signature

## Vectorized Implementation

The per-token loop above is for clarity. In practice, vectorize over all scored positions in a window:

```python
# Vectorized counter lookup + mixing for all positions in a batch
buckets = (x[:, scored_start:scored_end] % n_buckets).long()  # [B, stride]
P_bi_batch = count_bi[buckets.reshape(-1)].reshape(B, stride, V)
P_bi_batch = P_bi_batch / P_bi_batch.sum(dim=-1, keepdim=True)

H_batch = -(P_neural_scored * (P_neural_scored + 1e-10).log()).sum(dim=-1)
sigma_batch = torch.sigmoid(w_gate * H_batch + b_gate)
# ... rest follows same pattern
```

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ruled illegal by organizers | LOW (TTT precedent is stronger) | Fatal | Ask in Discord before submitting |
| EWMA counters too noisy early | MEDIUM | Reduces early-doc benefit | Initialize with training-set unigram frequencies |
| Hash collisions blur bigram signal | LOW | Reduces bigram benefit | Use prime-sized buckets, multiple hash functions |
| Evaluation too slow (per-token loop) | MEDIUM | Exceeds time budget | Vectorize (see above), batch-process windows |
| Marginal improvement on strong model | MEDIUM | Wasted effort | Cheap to test — zero training cost |

## References

- Grave et al. (2016). "Improving Neural Language Models with a Continuous Cache." arXiv:1612.04426.
- Khandelwal et al. (2020). "Generalization through Memorization: Nearest Neighbor Language Models." arXiv:1911.00172.
- Parameter Golf PR #549 (Legal TTT precedent): accepted SOTA at 1.1194 BPB.
- Parameter Golf PR #978 (Normalization analysis): properly normalized n-gram achieves only 1.51 BPB standalone.
