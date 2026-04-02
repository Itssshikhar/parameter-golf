# PR #1246: Trinity Ternary GPT

**BPB: 0.9650** (ternary roundtrip)
**Status:** OPEN
**Author:** deborahnelson8788726
**Date:** 2026-04-02

## Key Techniques

1. **BitNet b1.58 Ternary QAT** - Ternary weights {-1, 0, +1} from step 0
   - Absmean ternary quantization (per-group, group_size=128)
   - scale = mean(|w|), w_q = round(w/scale).clamp(-1,1)
   - Adapted from Trinity framework's ternary_pipeline.zig

2. **Base-3 Ternary Packing** - 5 trits per byte (3^5=243 < 256)
   - Extremely efficient storage: ~1.6 bits/param
   - Adapted from Trinity's ternary_packing.zig
   - Enables ~73M parameters to fit within 16MB limit

3. **NeoMuon Optimizer** - 3 Newton-Schulz steps (vs standard 5)
   - Faster per-step, allows more gradient updates in wallclock budget

4. **Z-loss Regularization** (1e-4) for stable logits with ternary STE
5. **relu^2 activation** with 4x MLP expansion (ternary weights are cheap, so go wide)

## Architecture
- 10 layers (not 11!), 768 model dim (wider than standard 512)
- 8 heads / 4 KV heads (GQA)
- relu^2 activation, 4x MLP expansion (3072 hidden)
- U-Net skip connections with learned skip weights
- Partial RoPE (16/96 dims)
- EMA (0.997 decay, starts at step 500)
- 524k batch tokens, seq_len=1024
- 1489 steps in 10 min on 8xH100 SXM
- 14.2MB artifact

## What's Novel
- **Ternary QAT from step 0** - most submissions quantize post-training
- **73M parameters at 16MB** - 2.7x more params than standard approaches
- **NeoMuon (3 NS steps)** - speed optimization for ternary training
- **Base-3 packing** - custom compression for ternary weights

## What's Borrowed
- Trinity framework concepts (ternary packing, quantization)
- BitNet b1.58 approach

## Compatibility with SOTA Stack
- NOT directly compatible - fundamentally different quantization approach
- The idea of training with 73M params and ternary quantizing to 16MB is orthogonal
- Could potentially combine ternary training with SLOT eval-time optimization
- NeoMuon (3 NS steps) could be used with standard training too
- No weight decay (incompatible with ternary STE)

## CRITICAL NOTES
- 0.9650 BPB is impressive but needs careful verification
- Ternary roundtrip means the score is after packing/unpacking
- Only 10 layers and 1489 steps - very different training regime
- The wider model (768d) compensates for information loss from ternary weights
- No TTT or SLOT mentioned - pure model quality
- LZMA preset=9 for final compression
