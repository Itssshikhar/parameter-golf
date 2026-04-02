# PR #1254: XSA + LoRA TTT

**BPB: 1.1070**
**Status:** OPEN
**Author:** Elarwei001 (Elar Wei)
**Date:** 2026-04-02

## Key Techniques

1. **LoRA TTT (Test-Time Training)** - The main differentiator
   - Adds ephemeral LoRA adapters (rank=8) to Q, V projections + LM head
   - Splits each document into 256-token chunks with 50% overlap
   - Process chunks left-to-right over 2 epochs
   - Score tokens on final epoch
   - Train LoRA on all chunks except the last one in final epoch
   - Massive improvement: 1.519 pre-TTT -> 1.1070 post-TTT (-27.1%)

2. **XSA (Exclusive Self Attention)** on all layers
3. **BPE-8192 tokenizer** (larger vocab than standard SP-1024)
4. **QAT Int6** quantization (enabled at 15% of training)
5. **Sliding window attention** (window_size=192)

## Architecture
- 11 layers, d_model=416 (smaller than standard 512), 8 heads, 4 KV heads (GQA)
- 3x MLP expansion with LeakyReLU(0.5)^2
- RMSNorm, RoPE
- Tied embeddings
- BPE-8192 vocabulary (8,192 tokens)
- ~20.5M parameters
- 14.4 MB compressed with int8 + zlib

## What's Novel
- **LoRA-based TTT** - lighter weight than full TTT, adapts only Q, V, and LM head
- **BPE-8192 tokenizer** - larger vocabulary allows more expressive tokens
- **Smaller d_model (416)** compensated by larger vocabulary

## What's Borrowed
- BPE-8192 from @sproos
- LoRA TTT concept from @LoquiAuris, @MatoTeziTanka (PR #548, #512)
- XSA from PR #198
- LeakyReLU(0.5)^2 from PR #549
- Int6 QAT from PR #414

## Compatibility with SOTA Stack
- LoRA TTT is an eval-time technique, compatible with any model
- BPE-8192 tokenizer requires different data preprocessing
- Smaller d_model (416 vs 512) is a design trade-off, not directly stackable
- The LoRA TTT approach could be applied to the standard SOTA stack

## Notes
- Pre-TTT BPB of 1.519 is much worse than SOTA (1.1147), showing the model relies heavily on TTT
- The -27.1% TTT improvement is very large, suggesting potential overfitting risk
- Training time ~8 min + TTT ~2 min = total ~10 min
