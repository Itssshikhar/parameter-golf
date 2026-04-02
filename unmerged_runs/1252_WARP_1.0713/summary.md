# PR #1252: WARP (Word-Aware Representation Priors) + Legal TTT + Context-Only SLOT

**BPB: 1.0713**
**Status:** CLOSED (likely due to 1xH100 training, not standard 8xH100)
**Author:** ahmetdenizyilmaz (Ahmet Deniz YILMAZ)
**Date:** 2026-04-02

## Key Techniques

1. **WARP (Word-Aware Representation Priors)** - The main innovation (183,820 params, 0.7% overhead)
   - **WARP-Len** (6,657 params): Word length embedding at layer 0. Each token gets embedding based on how many BPE tokens its word contains. Injected before RMSNorm.
   - **WARP-Pos** (1,035 params): Word position bias in Q and K. Learned per-layer embeddings based on within-word position (0-7). Shared across all 11 layers with per-layer learned scales.
   - **WARP-Type** (176,128 params): Word type logit bias at output. 2-layer classifier (512->192->64 types) produces soft type probabilities. Multiplied with learned type-vocabulary bias matrix (64x1024) and added to logits before softcap. No auxiliary loss.
   - All modules share `compute_word_boundary_maps()` - detects word starts from SentencePiece leading-space convention using only token IDs. Fully `torch.compile` compatible.

2. **EMA Disabled for Short Runs** - Single largest improvement
   - EMA (beta=0.997) degrades by +0.045 BPB at ~1260 steps
   - Crossover where EMA helps: ~3000+ steps
   - Set EMA_DECAY=0.0

3. **Context-Only SLOT** (lr=0.005, 8 steps)
4. **Legal TTT** (2 epochs, freeze_blocks=2, score-first)

## Architecture
- 11L, 512d, 8H/4KV GQA, LeakyReLU(0.5) squared 3x
- BigramHash 2816 buckets, XSA all 11 layers, SmearGate
- ValueEmbedding REMOVED (freed params for WARP-Type)
- ~27M params, GPTQ int6, 13.65 MB artifact

## What's Novel
- **WARP system** is genuinely novel - restores word boundary information lost by BPE tokenization
- **EMA-disable discovery** for short training runs
- The insight that BPE destroys word boundaries that models must re-learn through attention

## What's Borrowed
- Base architecture from PR #549 (@abaybektursun)
- TTT recipe from PR #461 (@Christopher-Lee-McClendon)
- SLOT from PR #1084 (@AnubhavBharadwaaj)

## Compatibility with SOTA Stack
- Highly compatible - WARP modules are small parameter additions
- Could be added to any transformer-based approach
- WARP-Type replaces ValueEmbedding (parameter trade-off)
- On 8xH100 with full compute budget, author estimates much better results (0.9766 in 50-min run)

## CRITICAL NOTES
- Only trained on 1xH100 (1260 steps) - would get ~7185 steps on 8xH100
- Author suspects there may be errors, welcomes verification
- The 50-min run hit 0.9766 but exceeded 16MB (16.81 MB)
- WARP idea is very promising but needs 8xH100 verification
