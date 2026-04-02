# PR #1260: MuonEq-R + Depth Recurrence + Mixed Int5/Int6 GPTQ

**BPB: 1.0929** (3-seed mean, std 0.0009)
**Status:** OPEN
**Author:** dexhunter (Dex)
**Date:** 2026-04-02

## Key Techniques

1. **MuonEq-R** (Row-normalized Muon)
   - Row-normalizes gradient matrices BEFORE Newton-Schulz orthogonalization
   - Improves conditioning of NS5 iteration for non-square weight matrices
   - Zero-byte cost, ~0.001 BPB improvement
   - Zero-parameter addition, pure optimizer improvement

2. **Depth Recurrence** - Layers 4,5 repeated
   - Virtual layers 12-13 on top of 11 physical layers
   - MLP weights fully shared during recurrence (REPEAT_UNTIE_MLP=none)
   - Zero extra parameters
   - Activated at step 3000 with 20-step linear warmup
   - ~0.003 BPB improvement

3. **Mixed Int5/Int6 GPTQ**
   - Hessian sensitivity ranking: 60 int6 + 6 int5 layers
   - Optimal size/quality tradeoff
   - Allows more aggressive compression on less-sensitive layers

## Architecture
- Same as PR #1218 (4096-Vocab + MLP 4x + WD 0.085)
- ~5538 steps at 106.6ms/step
- All seeds under 16MB (max: 15,981,324 bytes)
- No TTT, no SLOT, no eval-time adaptation

## What's Novel
- **MuonEq-R** - Simple but effective optimizer modification
- **Mixed quantization** - Different bit widths based on Hessian sensitivity
- **Depth recurrence warmup** - Activating recurrence mid-training with linear warmup

## What's Borrowed
- PR #1218 by @clarkkev (4096-Vocab + MLP 4x + WD 0.085 foundation)
- PR #1019 by @abaybektursun (GPTQ + XSA + BigramHash baseline)
- PR #1204 by @msisovic (depth recurrence concept)

## Compatibility with SOTA Stack
- **Highly compatible** - all three techniques are incremental improvements
- MuonEq-R is a drop-in optimizer replacement (zero-cost)
- Depth recurrence adds computation but no parameters
- Mixed quantization is a compression improvement
- Could combine with TTT/SLOT for additional gains
- No TTT used here means there's significant headroom with eval-time techniques

## CRITICAL NOTES
- Achieves 1.0929 WITHOUT ANY eval-time adaptation (no TTT, no SLOT)
- This is important: the pure model quality before TTT/SLOT is very strong
- Adding SLOT (which gave -0.191 BPB in PR #1263) could theoretically push this much lower
- The compressed code format (base85-encoded) makes it harder to inspect
- Clean, legal, fully reproducible (3-seed verification)
