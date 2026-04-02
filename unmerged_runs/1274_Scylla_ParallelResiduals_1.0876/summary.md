# PR #1274: Scylla + Parallel Residuals + Mini Depth Recurrence + Legal TTT

**BPB: 1.0876** (3-seed mean, std 0.00037)
**Status:** OPEN
**Author:** MatoTeziTanka (Norelec) - same team as merged SOTA (PR #1019)
**Date:** 2026-04-02

## Key Techniques

1. **Scylla Tokenizer** - 998-token TokenMonster vocabulary (PR #1143, @simon-marcus)
   - WARNING: PR #1271 (Scylla Byte Audit) showed sub-1.0 Scylla claims were measurement errors
   - This PR likely uses corrected byte accounting (claims 1.0876, not sub-1.0)

2. **Parallel Residuals** - From layer 7, learned 4-scalar routing (PR #1204, @msisovic)
   - Allows multiple residual paths through later layers
   - Learned routing scalars determine path weighting

3. **Mini Depth Recurrence** - Layers 4,5 repeated, untied MLPs (PR #1204)
   - Zero extra params for attention (shared weights)
   - MLPs can be untied for more capacity

4. **Legal Score-First TTT** - SGD, lr=0.005, 3 epochs (their PR #549)
5. **Mixed INT5/INT6 quantization** + brotli-11 compression
6. **N-gram two-pass rescoring** - Reported separately, NOT used as submission metric

## Architecture
- 11L, 512d, GQA(8/4), MLP 3x, LeakyReLU(0.5)^2
- XSA last 4 layers, SmearGate, BigramHash(2048, 128)
- EMA + SWA, Parallel Muon
- ~5879 steps at 102.1ms/step

## What's Novel
- **Combination of parallel residuals + depth recurrence** on Scylla tokenizer
- From their own prior record work (PRs #399, #549, #1019)
- Beats merged SOTA by -0.0271 BPB (Welch t = -91.92, p << 0.01)

## What's Borrowed
- Scylla tokenizer from @simon-marcus (PR #1143)
- Parallel residuals + depth recurrence from @msisovic (PR #1204)
- Legal TTT from @Christopher-Lee-McClendon (PR #461)
- Mixed quantization concept from PR #1105

## Compatibility with SOTA Stack
- Highly compatible - this IS the SOTA team improving their own stack
- Parallel residuals and depth recurrence are architecture changes
- Scylla tokenizer requires retokenization of data
- Mixed INT5/INT6 is a compression improvement
- Could potentially combine with SLOT for even lower BPB

## CRITICAL NOTES
- The Scylla tokenizer byte accounting concern (PR #1271) needs to be verified
- If byte accounting is correct, this represents a genuine -0.027 BPB improvement
- The authors are the merged SOTA holders (@MatoTeziTanka, same as PR #1019)
- N-gram rescoring gives additional -0.195 BPB but is reported separately
