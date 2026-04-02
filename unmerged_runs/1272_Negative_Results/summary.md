# PR #1272: Comprehensive Negative Results -- What Doesn't Work on Strong Models

**Status:** OPEN (research/informational, not a submission)
**Author:** andrewbaggio1 (andreww)
**Date:** 2026-04-02

## Key Findings: What DOESN'T Work on Well-Trained GPTQ'd Models

| Technique | BPP delta | Why it fails |
|-----------|:---------:|:-------------|
| N-gram (Kneser-Ney, exact trie) | +0.001 to -0.003 | Model is 100x better than n-gram. Mixing dilutes confidence. |
| Online Logit Bias (per-token SGD) | +0.003 (hurts) | GPTQ'd model already well-calibrated. Takes 1229s. |
| Prime MLP Adapters (rank-64) | -0.00009 | Only helps weak baselines (1.50 BPP). 1.11 baseline has no room. |
| Complementary Training | -0.0004 (noise) | Model already knows everything bigram knows at convergence. |
| Score-first chunked TTT | -0.003 | Tiny gain on GPTQ'd models. |

## Critical: N-gram Normalization Proof

The 0.09-0.97 BPP "improvement" from hashed n-gram caches was a measurement
artifact from unnormalized distributions. Properly normalized n-grams give
only 0.001-0.003 BPP improvement.

## Critical: SLOT Violates Causal Dependence

- 100% violation rate across 240 tested pairs
- Self-prediction advantage: +0.24 nats (shared delta), +0.73 nats (per-sample)
- Every SLOT-based result on the leaderboard is suspect

## Impact for Our Work

1. Don't waste time on n-gram rescoring -- it's noise on strong models
2. TTT gives tiny gains (-0.003) on GPTQ'd models -- diminishing returns
3. SLOT gives massive gains but may be ruled illegal
4. Focus on model quality improvements (architecture, optimizer, quantization)
5. The frontier is now about architecture changes, not eval tricks
