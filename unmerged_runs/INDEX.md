# Unmerged Parameter Golf PRs -- Study Index

**Generated:** 2026-04-03
**Merged SOTA:** 1.1147 BPB (PR #1019, @abaybektursun)
**Context:** 16MB model, 10 min training on 8xH100, BPB metric

## Summary Table

| PR | BPB | Status | Key Technique | Credibility | Compatibility w/ SOTA |
|----|-----|--------|---------------|-------------|----------------------|
| **#1263** | **0.9354** | OPEN | SLOT (16 AdamW steps on delta+logit_bias) | HIGH but SLOT legality questioned | DROP-IN (eval-time only) |
| **#1246** | **0.9650** | OPEN | Ternary QAT (BitNet b1.58, 73M params in 16MB) | MEDIUM (needs verification) | INCOMPATIBLE (different paradigm) |
| **#1241** | **0.9901** | OPEN | MDLM Diffusion + EOS learning | MEDIUM (non-standard metric, non-compliant HW) | INCOMPATIBLE (diffusion, not AR) |
| **#1252** | **1.0713** | CLOSED | WARP (word boundary info) + SLOT | MEDIUM (1xH100, author unsure) | HIGH (WARP modules are small) |
| **#1257** | **1.0855** | OPEN | Complement Training + TTT | MEDIUM (8xA100, 3600s budget) | MEDIUM (complement training is additive) |
| **#1274** | **1.0876** | OPEN | Scylla + Parallel Residuals + Depth Recurrence | HIGH (same team as SOTA, 3-seed) | HIGH (incremental on SOTA) |
| **#1260** | **1.0929** | OPEN | MuonEq-R + Depth Recurrence + Mixed Quant | HIGH (3-seed, no TTT/SLOT) | VERY HIGH (all techniques are drop-in) |
| **#1254** | **1.1070** | OPEN | XSA + LoRA TTT | MEDIUM (single seed) | MEDIUM (LoRA TTT is eval-time) |

## Investigation/Research PRs

| PR | Topic | Key Finding |
|----|-------|-------------|
| **#1271** | Scylla Byte Audit | Sub-1.0 Scylla claims were measurement errors (93% of gap = byte accounting) |
| **#1272** | Negative Results | N-gram rescoring is noise on strong models; TTT gives -0.003; SLOT may be illegal |
| **#1240** | SLOT Causal Dependence | 100% violation rate; SLOT uses future information |

## Technique Rankings (by reliability and applicability)

### Tier 1: High-confidence, compatible with SOTA stack
1. **MuonEq-R** (PR #1260) -- Zero-cost optimizer improvement. Row-normalize gradients before NS5. ~0.001 BPB.
2. **Depth Recurrence** (PR #1260, #1274) -- Repeat layers 4,5. Zero extra params, ~0.003 BPB.
3. **Mixed Int5/Int6 GPTQ** (PR #1260) -- Hessian sensitivity ranking. Better compression.
4. **Parallel Residuals** (PR #1274) -- Learned 4-scalar routing from layer 7. Architecture change.
5. **QK-Gain 4.0** (PR #1263) -- Per-head gain initialized at 4.0. Simple hyperparameter.

### Tier 2: Promising but needs verification or has caveats
6. **SLOT** (PR #1263) -- -0.191 BPB improvement but legality questioned. If legal, this is game-changing.
7. **WARP** (PR #1252) -- Novel word boundary injection. Only tested on 1xH100.
8. **LoRA TTT** (PR #1254) -- Lighter than full TTT but massive pre-TTT degradation.
9. **Complement Training** (PR #1257) -- Needs validation on 8xH100 @ 600s.

### Tier 3: Different paradigm / not directly applicable
10. **Ternary QAT** (PR #1246) -- 73M params in 16MB. Fundamentally different approach.
11. **MDLM Diffusion** (PR #1241) -- Non-AR paradigm. Different metric.

## Key Takeaways

1. **SLOT is the biggest single-technique improvement** (-0.191 BPB) but may be ruled illegal
2. **Without eval-time tricks, the frontier is ~1.09 BPB** (PR #1260: MuonEq-R + Depth Recurrence, no TTT/SLOT)
3. **PR #1271 and #1272 debunk several techniques**: n-gram rescoring is noise, Scylla byte accounting was buggy, TTT gives tiny gains on strong models
4. **Most compatible techniques for immediate use**: MuonEq-R, Depth Recurrence, Mixed Quantization, QK-Gain 4.0
5. **WARP is the most interesting novel architecture idea** but needs 8xH100 validation

## Files in Each Directory

Each PR directory contains:
- `README.md` -- Original PR README
- `train_gpt.py` (or `train_mdlm.py`, `train_sota1006.py`) -- Full training script
- `summary.md` -- Analysis with BPB, techniques, novelty, compatibility assessment
- Key extracted code files (e.g., `slot_implementation.py`, `warp_implementation.py`)

## Methodology Notes

- PRs were fetched on 2026-04-03 from `openai/parameter-golf` (state=open)
- PR #1252 is CLOSED (was open when originally identified)
- PR #1242 is CLOSED (Scylla + n-gram, 1.0903 BPB, superseded by #1274)
- Code was extracted from `gh pr diff` output
- PR #1260's train_gpt.py is base85-encoded/lzma-compressed (self-extracting)
