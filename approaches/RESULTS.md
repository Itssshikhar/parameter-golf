# Experimental Results

## Run Config: 1×H100, 5 min training, SOTA stack
Branch: `approach-geometry-pipeline`
Config: 1×H100 80GB HBM3, 300s wallclock, batch=65536, seq=1024, warmdown=600, seed=1337

## Results Table

| # | Experiment | Steps | Step Avg (ms) | Params | val_bpb (s64) | Delta | Artifact |
|---|---|---|---|---|---|---|---|
| 1 | **BASELINE** (SOTA, all novel off) | 2501 | 120.0 | 27.0M | **1.3571** | — | 15.57MB |
| 2 | **+GNS** (Gram-Newton-Schulz) | 2543 | 118.0 | 27.0M | **1.3550** | **-0.0021** | 15.67MB |
| 3 | **+GNS+MTP** (Multi-Token Prediction K=3) | 3005 | 99.85 | 28.6M* | **1.3430** | **-0.0141** | 15.82MB |
| 4 | +GNS+MTP+BPB loss | — | — | — | — | — | (running) |
| 5 | Low-Rank Deep (18L, rank=128) | — | — | — | — | — | (pending) |
| 6 | Meta-Warmdown + TTT | — | — | — | — | — | (pending) |

*MTP heads (1.6M params) are stripped at export — artifact includes only the base 27M params.

## Key Findings

### GNS (Approach #5): Modest throughput improvement
- +42 steps (+1.7%) from fused polynomial NS iteration
- -0.0021 BPB — small but free (no quality tradeoff)

### MTP (Approach #4): Major training quality improvement
- +504 steps (+20%) — the MTP auxiliary heads appear to be FASTER than the baseline (99.85ms vs 120ms per step). This is unexpected and may be due to torch.compile optimizing the multi-head loss computation.
- -0.0141 BPB total (including step count improvement)
- MTP provides 4× gradient supervision per token position — the model learns faster per step
- The MTP heads are free at inference (stripped from export)

### Combined GNS+MTP vs Baseline
- **20% more training steps** (3005 vs 2501)
- **-0.0141 BPP improvement** (-1.04%)
- Zero additional artifact cost (MTP heads stripped)
- Zero architecture change at inference

## Budget Tracking

| Run | Experiment | H100 Time | Est. Cost |
|---|---|---|---|
| 1 | Baseline only (first attempt) | ~22 min | ~$1.30 |
| 2 | FA3 crash (wasted) | ~1 min | ~$0.10 |
| 3 | Baseline + GNS (MTP timed out) | ~51 min | ~$3.00 |
| 4 | GNS+MTP single experiment | ~28 min | ~$1.60 |
| 5 | GNS+MTP+BPB (running) | ~28 min | ~$1.60 |
| **Total** | | | **~$7.60** |
| **Remaining** | | | **~$22.40** |

## Scaling Estimate to 8×H100 / 10 min

At full competition config (8×H100, 10 min, batch=786K):
- Baseline SOTA: ~7000 steps → 1.1147 BPB
- With GNS+MTP: ~8400 steps (+20%) → estimated **1.10-1.11 BPB**

The relative improvement (-1.04%) should hold. MTP's gradient quality improvement may be even larger with more training steps (auxiliary losses become more useful as the model improves).
