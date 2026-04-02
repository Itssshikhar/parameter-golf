# PR #1241: MDLM Diffusion + EOS Learning + Full Dataset Shard Rotation

**BPB: 0.9901** (variational ELBO, 128 eval steps)
**Status:** OPEN (non-record: 1x AWS A10G, not 8xH100)
**Author:** aiejvn
**Date:** 2026-04-02

## Key Techniques

1. **MDLM (Masked Diffusion Language Model)** - Completely different paradigm from AR models
   - Bidirectional transformer (is_causal=False)
   - Log-linear noise schedule: alpha(t) = 1 - (1-eps)*t
   - Discrete absorbing-mask ELBO evaluation
   - Antithetic time sampling for variance reduction
   - AdaLN timestep conditioning (log-sigma -> scale+shift per layer)

2. **EOS Token Learning** (novel for diffusion LMs)
   - Token 1 (<s> in SP1024) marks document boundaries
   - EOS positions NEVER masked during diffusion - serve as structural anchors
   - Dedicated PAD_ID=1025 (separate from MASK_ID=1024) fills post-EOS positions
   - PAD excluded from loss via content_mask
   - Separating PAD from MASK prevents collision between structural padding and diffusion masking

3. **Shard Rotation** for memory-constrained training
   - ShardedDataLoader loads N shards at a time, rotates between groups
   - Enables full FineWeb 10B training without loading entire dataset into RAM
   - Pre-allocated buffer to avoid 2x peak allocation

4. **Ablation: Head count invariance** for diffusion LMs
   - Val BPB flat across {2, 4, 8, 16, 32} heads at fixed model dim
   - Different from AR models where head count matters

## Architecture
- 11 layers, 512 dim, 8 heads, MLP 3x (ReLU^2), RoPE
- TOTAL_VOCAB = 1026 (1024 real + MASK + PAD), embedding table padded to 1088
- AdamW: lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1
- 33M parameters

## What's Novel
- **First diffusion LM with EOS learning** in this competition
- **Shard rotation** for memory-efficient training on consumer hardware
- **Head count invariance** finding for bidirectional diffusion LMs
- Achieves sub-1.0 BPB with diffusion (0.9901 vs 1.1465 for prior best diffusion)

## What's Borrowed
- MDLM framework from PR #1106
- Discrete ELBO evaluation from MDLM paper (Sahoo et al., 2024)
- Architecture inspired by LLaDA (Nie et al., 2025)

## Compatibility with SOTA Stack
- NOT compatible - fundamentally different paradigm (diffusion vs autoregressive)
- Cannot be combined with TTT, SLOT, n-gram rescoring, or other AR eval techniques
- Uses different training loss, evaluation metric, and model architecture
- But diffusion LM approaches could become the next paradigm shift

## CRITICAL NOTES
- Trained on 1x AWS A10G for 1267 min (~21 hours) - NOT 8xH100 compliant
- The 0.9901 BPB uses variational ELBO (128 eval steps) - different metric from AR val_bpb
- PR #1271 (Scylla Byte Audit) showed sub-1.0 claims can be measurement errors
- Needs 8xH100 rerun for wall-clock compliance
- The BPB metric for diffusion LMs may not be directly comparable to AR models
