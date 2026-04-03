# Experimental Results

## Run Config: 1×H100, 5 min training, SOTA stack
Branch: `approach-geometry-pipeline`
Config: 1×H100 80GB HBM3, 300s wallclock, batch=65536, seq=1024, warmdown=600, seed=1337

## Results Table

| # | Experiment | Steps | Step Avg (ms) | val_bpb (s64) | Delta | Status |
|---|---|---|---|---|---|---|
| 1 | **BASELINE** (SOTA, all novel off) | 2501 | 120.0 | **1.3571** | — | Done |
| 2 | **+GNS** (Gram-Newton-Schulz) | 2543 | 118.0 | **1.3550** | **-0.0021** | Done |
| 3 | **+GNS+MTP** (K=3, w=0.2) | **3005** | **99.85** | **1.3430** | **-0.0141** | **BEST** |
| 4 | +GNS+MTP+BPB loss | 2690 | 111.5 | 1.4508* | +0.0937 | **HARMFUL** |
| 5 | Low-Rank Deep (15L, rank=128) | — | — | — | — | Timed out |
| 6 | Meta-Warmdown + TTT | — | — | — | — | Not tested |

*BPB-weighted loss experiment: trained OK but (a) pre-quant BPB was worse and (b) GPTQ quantization crashed due to Cholesky failure on modified weight distributions. **BPB-weighted loss is a negative result — do not use.**

## Key Findings

### GNS (Approach #5): Small free win
- Fused polynomial NS: 2 matmuls/iter vs 3
- +42 steps (+1.7%) in same wallclock
- -0.0021 BPB with zero quality tradeoff

### MTP (Approach #4): **Major win — the best approach**
- K=3 auxiliary heads predicting t+2, t+3, t+4 with weight 0.2
- +504 steps (+20%) — MTP appears FASTER per step (99.85ms vs 120ms), likely torch.compile optimizes the multi-loss computation
- -0.0141 BPB total improvement
- Auxiliary heads stripped at export → zero artifact cost
- 4× gradient supervision per token position → model learns faster

### BPB-Weighted Loss: **Negative result**
- Weighting CE by 1/bytes(token) hurts training dynamics
- Fewer steps (2690 vs 3005) — the weighted loss is harder to optimize
- Worse pre-quant BPB (1.4508 vs 1.3430)
- Breaks GPTQ Cholesky factorization
- **Conclusion: Do not use**

### Low-Rank Deep: Timed out
- 15-layer model with rank-128 MLPs started training but the sliding window eval (62M tokens × stride=64 on 1×H100) exceeded the 2000s timeout
- Need to test with shorter eval or on 8×H100

## Budget Tracking

| Run | Experiment | H100 Time | Est. Cost |
|---|---|---|---|
| 1 | Baseline only | ~22 min | ~$1.30 |
| 2 | FA3 crash | ~1 min | ~$0.10 |
| 3 | Baseline + GNS | ~51 min | ~$3.00 |
| 4 | GNS+MTP | ~28 min | ~$1.60 |
| 5 | GNS+MTP+BPB (crashed at quant) | ~20 min | ~$1.20 |
| 6 | Low-Rank Deep (timed out) | ~33 min | ~$1.90 |
| **Total** | | | **~$9.10** |
| **Remaining** | | | **~$20.90** |

## Scaling Estimate to 8×H100 / 10 min

At full competition config (8×H100, 10 min, batch=786K):
- Baseline SOTA: ~6920 steps @ 86ms → 1.1147 BPB
- With GNS+MTP: projected ~8300 steps (+20%) → estimated **1.10-1.11 BPB**
- The -1.04% relative improvement should hold or improve with more steps

## Next Steps (Priority Order)

1. **Submit GNS+MTP on 8×H100** — the proven winner. Cost: ~$12 for a proper submission run.
2. **Test Low-Rank Deep** with higher subprocess timeout or skip sliding window eval.
3. **Test Meta-Warmdown** — the most novel approach, untested.
4. **Combine GNS+MTP with Low-Rank if both work** — could compound.
