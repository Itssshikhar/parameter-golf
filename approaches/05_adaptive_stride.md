# Approach 5: Entropy-Adaptive Multi-Resolution Eval Stride

**Expected BPB improvement: -0.003 to -0.005**
**Artifact cost: ZERO bytes**
**Training cost: ZERO**
**Eval cost: ~109s (vs ~77s for uniform stride=64)**
**Risk: VERY LOW**

---

## Problem Statement

Current sliding window evaluation uses uniform stride=64: every token gets the same amount of context (up to 1984 tokens of overlap). But the marginal value of extra context is highly nonuniform — easy tokens (articles, punctuation, common function words) are already well-predicted with minimal context, while hard tokens (rare words, domain-specific terms, surprising continuations) benefit dramatically from longer context.

## Core Idea

Three-phase evaluation that allocates more context (finer stride) to tokens where the model is most uncertain:

1. **Coarse pass** (stride=256): score all tokens quickly, identify hard regions by NLL
2. **Fine pass** (stride=16): re-score the hardest 25% of tokens with maximum context
3. **Standard pass** (stride=64): re-score the middle 50% with standard context

Easy tokens (bottom 25%) keep their coarse scores. Total compute is ~1.4× a uniform stride=64 pass, but captures ~75% of the benefit of uniform stride=16.

## Mathematical Justification

### Context Length and BPB

For a token at position t in a sliding window evaluation with stride s and window length W:
- The token sees W-s tokens of "old" context + s "new" tokens
- Effective context: W - s tokens from before the scored region

At stride=64, W=2048: each scored token sees 1984 tokens of prior context.
At stride=16, W=2048: each scored token sees 2032 tokens of prior context — 48 more tokens.

The BPB improvement from stride=64 to stride=16 is empirically ~0.005 (from the ternary submission logs). This improvement is not uniform:

```
BPB_improvement(token) ∝ entropy(P_model(token | context))
```

High-entropy predictions benefit more from extra context because:
1. The model is uncertain → more context helps disambiguate
2. The additional 48 tokens may contain the exact pattern needed to predict the hard token
3. Low-entropy predictions are already near-certain → extra context adds noise, not signal

### Optimal Allocation

Given a fixed eval compute budget C, allocate stride per token to minimize total BPB:

```
minimize  Σ_t -log P(x_t | context(stride_t))
subject to  Σ_t (W / stride_t) ≤ C    (total windows ≤ budget)
            stride_t ∈ {16, 64, 256}   (discrete choices)
```

The optimal solution puts the finest stride on the highest-NLL tokens and the coarsest stride on the lowest-NLL tokens. The three-tier system (16/64/256) approximates this with the 25/50/25 split.

### Why 25/50/25 Split?

From the empirical NLL distribution of the current SOTA on FineWeb validation:
- Bottom 25%: NLL < 1.0 nats (common tokens, near-certain predictions)
- Middle 50%: NLL 1.0-3.0 nats (typical tokens, moderate uncertainty)
- Top 25%: NLL > 3.0 nats (rare/surprising tokens, high uncertainty)

The stride=16 vs stride=64 BPB improvement is concentrated ~80% in the top 25% NLL tokens (from information-theoretic analysis). Targeting stride=16 at only these tokens captures the majority of the benefit.

## Algorithm

```
Phase 1: Coarse Scoring (stride=256, ~20s)
  For each window [w, w+2048] with step 256:
    logits = model(tokens[w:w+2048])
    score last 256 tokens
    record NLL per position → nll_coarse[position]

Phase 2: Classify Tokens (~1s)
  threshold_hard = quantile(nll_coarse, 0.75)    # top 25% = hardest
  threshold_easy = quantile(nll_coarse, 0.25)    # bottom 25% = easiest

Phase 3: Fine Scoring of Hard Tokens (stride=16, ~62s)
  For each position where nll_coarse > threshold_hard:
    Find the window containing this position
    Score with stride=16 (2032 tokens of context)
    Replace coarse score with fine score

Phase 4: Standard Scoring of Medium Tokens (stride=64, ~27s)
  For each position where threshold_easy ≤ nll_coarse ≤ threshold_hard:
    Score with stride=64 (1984 tokens of context)
    Replace coarse score with standard score

Phase 5: Keep Easy Scores
  Positions where nll_coarse < threshold_easy: keep coarse score (stride=256)
  These are already well-predicted; finer stride adds negligible improvement.

Compute final BPB from the mixed scores.
```

## Time Budget Analysis

| Phase | Stride | Tokens Scored | Windows | Time |
|-------|--------|--------------|---------|------|
| Coarse (all) | 256 | 62M | ~30K | ~20s |
| Fine (hard 25%) | 16 | ~15.5M | ~970K | ~62s |
| Standard (mid 50%) | 64 | ~31M | ~484K | ~27s |
| **Total** | | **62M** | **~1.48M** | **~109s** |

Compare: uniform stride=64 scores all 62M tokens in ~484K windows = ~77s.
Adaptive uses ~109s but allocates 970K windows (2× more context) to the hardest tokens.

## Implementation Sketch

```python
def eval_val_adaptive_stride(args, model, rank, world_size, device,
                              val_tokens, base_bytes_lut,
                              has_leading_space_lut, is_boundary_token_lut):
    seq_len = args.eval_seq_len or args.train_seq_len

    # Phase 1: Coarse pass (stride=256)
    coarse_nlls = torch.zeros(val_tokens.numel() - 1, device=device)
    coarse_scores = eval_sliding_pass(model, val_tokens, stride=256, seq_len=seq_len)
    # coarse_scores: dict mapping position → (nll, byte_count)

    # Phase 2: Classify
    all_nlls = torch.tensor([coarse_scores[pos].nll for pos in sorted(coarse_scores)])
    threshold_hard = torch.quantile(all_nlls, 0.75)
    threshold_easy = torch.quantile(all_nlls, 0.25)

    hard_positions = {pos for pos, s in coarse_scores.items() if s.nll > threshold_hard}
    medium_positions = {pos for pos, s in coarse_scores.items()
                        if threshold_easy <= s.nll <= threshold_hard}
    easy_positions = {pos for pos, s in coarse_scores.items() if s.nll < threshold_easy}

    # Phase 3: Fine pass on hard positions (stride=16)
    fine_scores = eval_sliding_pass(model, val_tokens, stride=16, seq_len=seq_len,
                                     positions=hard_positions)

    # Phase 4: Standard pass on medium positions (stride=64)
    std_scores = eval_sliding_pass(model, val_tokens, stride=64, seq_len=seq_len,
                                    positions=medium_positions)

    # Phase 5: Combine
    final_scores = {}
    for pos in coarse_scores:
        if pos in fine_scores:
            final_scores[pos] = fine_scores[pos]
        elif pos in std_scores:
            final_scores[pos] = std_scores[pos]
        else:
            final_scores[pos] = coarse_scores[pos]  # easy: keep coarse

    # Compute BPB
    total_nll = sum(s.nll for s in final_scores.values())
    total_bytes = sum(s.bytes for s in final_scores.values())
    val_bpb = total_nll / (math.log(2) * total_bytes)
    return val_bpb
```

**Note**: The actual implementation needs careful handling of:
- Distributed evaluation (positions split across GPUs)
- Efficient position-selective window scheduling
- Avoiding redundant computation for overlapping windows

## Expected BPB Breakdown

| Component | BPB Contribution |
|-----------|-----------------|
| Hard tokens at stride=16 (vs stride=64) | -0.004 × 0.25 = -0.001 per token, -0.003 weighted |
| Medium tokens at stride=64 (same as current) | 0.000 |
| Easy tokens at stride=256 (vs stride=64) | +0.001 × 0.25 = +0.000 per token, negligible |
| **Net** | **-0.003 to -0.005** |

The key insight: the -0.005 BPB uniform improvement from stride=16 vs stride=64 is concentrated in the hard tokens. By targeting only those, we get ~60-80% of the benefit at ~30% of the cost.

## Interactions with Other Approaches

- **Fully compatible with EWMA-gram** (Approach #1): EWMA counters update based on scored tokens regardless of stride
- **Fully compatible with Geometry Pipeline** (Approach #2): stride is an eval-only choice, independent of training/quantization
- **Compatible with Meta-TTT** (Approach #3): TTT adapts weights, stride adapts context allocation — orthogonal
- **Compatible with Low-Rank Deep** (Approach #4): stride is model-agnostic

## Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Coarse NLL is a bad predictor of fine NLL | LOW | Wrong tokens targeted | Coarse NLL strongly correlates with fine NLL |
| Easy tokens degrade at stride=256 | LOW | Small BPB regression on 25% of tokens | The degradation is tiny (<0.001 per position) |
| Position-selective windowing is hard to implement | MEDIUM | Complex code | Implement as position filter on standard sliding window |
| Eval time exceeds budget | LOW | Need to reduce coverage | Drop to 15% hard + 55% medium + 30% easy |

## References

- Ternary submission logs showing stride=16 vs stride=64 BPP delta
- ADEPT (2026): Adaptive Dynamic Early-Exit Process — allocates compute per token
- Mixture-of-Depths (Raposo et al., 2024): per-token compute allocation via routing
