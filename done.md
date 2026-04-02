# Parameter Golf — What Has Been Done (Top 20 Submissions)

## #1 — LeakyReLU² + Legal Score-First TTT + Parallel Muon (1.1194 BPB)
**Author:** abaybektursun | **Date:** Mar 23 | **Params:** 26.9M | **Artifact:** 15.98MB | **Steps:** ~7,185 @ 83.4ms

### Architecture
- 11 layers, 512d, 8 heads / 4 KV heads (GQA), 3× MLP (1536 hidden)
- **LeakyReLU(0.5)²**: `leaky_relu(x, 0.5).square()` instead of `relu(x).square()`. Preserves negative gradient flow, eliminates dead neurons. **-0.003 BPB**.
- **XSA (Cross-Scale Attention)** on last 4 layers: subtracts self-value projection from attention output via GQA-aware reshape (zero extra params, ~2ms overhead)
- **Partial RoPE**: only 16/64 head dims get rotary embeddings; remaining 48 dims position-agnostic
- **BigramHash(1536)**: XOR hash `(36313*curr) ^ (27191*prev) % 1535`, dim=128→512, scale=0.05
- **Value Embedding** on layers 9,10 only (dim=128, scale=0.1)
- **LN Scale Factor**: `1/sqrt(layer_idx+1)` — deeper layers get smaller norm outputs
- SmearGate, U-Net skips (5 enc + 6 dec), tied embeddings, logit softcap=30

### Training
- **Parallel Muon**: Parameter banking (4 contiguous 3D banks: qo, kv, mlp_up, mlp_down). 3-phase overlapped: async reduce-scatter → Adam step (overlapped) → NS5 + async all-gather. **-2% latency** (85→83ms), zero quality change.
- Muon: lr=0.025, momentum 0.92→0.99 over 1500 steps, WD=0.04, NS steps=5
- Adam: tied_embed_lr=0.035, scalar_lr=0.025, beta1=0.9, beta2=0.95, WD=0.04
- Batch: 786K tokens, seq=2048, warmdown=3500, grad_clip=0.3
- **EMA** (decay=0.997) every step + SWA (every 50 steps when scale<0.2)
- **Late QAT** at threshold 0.15 (STE fake int6 quantization in final ~520 steps)

### Test-Time Training (THE key differentiator)
- **Score-first protocol**: sliding window eval (stride=64) scores tokens in `torch.inference_mode()` → then SGD trains on already-scored chunks
- Chunks: 1,893 × 32K tokens, 3 epochs each, SGD+momentum=0.9, lr=0.002 with cosine decay
- **All blocks unfrozen** (freeze=0), grad_clip=1.0
- TTT gain: **-0.0025 BPB** (410s eval time)
- Total eval: ~530s (120s scoring + 410s TTT)

### Quantization
- GPTQ-lite int6: searches 5 clip percentiles per row (0.999, 0.9995, 0.9999, 0.99999, 1.0), picks min MSE
- LZMA preset=6 compression
- 3-seed mean: 1.1194 (std 0.0006)

### Ablations
| Change | BPB Impact |
|--------|-----------|
| LeakyReLU(0.5)² | -0.0021 (post-TTT) |
| Parallel Muon | ±0.0000 (2% faster) |
| TTT freeze=0 vs freeze=2 | -0.0004 |
| BigramHash 2048→1536 | -0.0009 |
| TTT itself | -0.0025 |

---

## #2 — 11L EMA + GPTQ-lite + warmdown3500 (1.1228 BPB)
**Author:** signalrush | **Date:** Mar 22 | **Params:** 27.0M | **Artifact:** 15.56MB | **Steps:** ~7,100 @ 84.5ms

### Architecture
- 11 layers, 512d, GQA (8/4), 3× MLP, XSA on last 4 layers, Partial RoPE (16/64)
- LN Scale Factor, Value Embedding (layers 9,10), SmearGate, BigramHash(2048)
- U-Net (5+6), tied embeddings, logit softcap=30

### Key Innovations (over PR #374 base at 1.1246)
- **GPTQ-lite clip search**: 5 candidate percentiles per weight row, pick min reconstruction MSE. **-0.0006 BPB**, zero training cost.
- **EMA (decay=0.997)**: every-step exponential moving average. Applied after training, before quantization. Stacks with SWA. **-0.0006 BPB**.
- **warmdown=3500** (from 3000): 30 extra seconds of low-LR training. **-0.0002 BPB**.
- **Late QAT threshold=0.15** (from 0.1): QAT kicks in ~500 steps earlier. **-0.0001 BPB**.

### Training
- Muon: lr=0.025, momentum 0.92→0.99/1500, WD=0.04, NS=5
- Adam: tied_embed_lr=0.035, scalar_lr=0.025, WD=0.04
- Batch: 786K, seq=2048, grad_clip=0.3

### Quantization
- Mixed int6 (MLP+attn) + int8 (embeddings) + fp16/fp32 (controls)
- zstd-22 compression
- 3-seed: 1.1228, 1.1236, 1.1236 (mean 1.1233, std 0.0005)

---

## #3 — 11L Partial RoPE + LN Scale + EMA + XSA4 (1.1248 BPB)
**Author:** jfprincz | **Date:** Mar 21 | **Params:** 26.8M | **Artifact:** 15.61MB | **Steps:** ~7,051 @ 85ms

### Key Innovations (over #4 at 1.1271)
- **Partial RoPE (16/64 dims)**: only first 16 of 64 head dims get rotary embeddings. 48 dims attend position-free. Reduces spurious position-content correlations. **-0.0023 BPB combined with LN Scale**.
- **Layerwise LN Scale**: `1/sqrt(layer_idx+1)`. Deeper layers have proportionally smaller outputs. Stabilizes deep residual networks.
- **Late QAT**: intended but was actually **dead code** — `torch.compile` constant-folded the flag away. Zero actual effect.

### Architecture
- 11L, 512d, GQA (8/4), 3× MLP, XSA last 4 layers
- EMA (0.997), SmearGate, BigramHash(2048), U-Net (5+6)
- FlashAttention 3, NTK-aware RoPE

### Quantization
- Int6 per-row (MLP+attn) + int8 (embeddings) + zstd-22
- 3-seed: 1.1248, 1.1250, 1.1253 (mean 1.1250, std 0.0005)

---

## #4 — 11L XSA4 + EMA + Int6 MLP3x (1.1271 BPB)
**Author:** jfprincz | **Date:** Mar 20 | **Params:** 26.8M | **Artifact:** 15.53MB | **Steps:** ~7,103 @ 84.5ms

### Key Innovations (over PR #198 at 1.1318)
- **XSA (Exclusive Self Attention)** on last 4 layers: subtracts normalized self-value projection from attention output. Forces layers to attend to context, not self. GQA-aware efficient reshape (zero allocation). **~-0.003 BPB**.
- **EMA (decay=0.997)** replacing SWA: continuous every-step averaging vs discrete checkpoint averaging. **~-0.002 BPB**.

### Architecture
- 11L, 512d, GQA, 3× MLP (relu²), OrthoInit with muP scaling
- SmearGate, BigramHash(2048, dim=128), U-Net (5+6)
- FlashAttention 3, NTK RoPE
- Muon: lr=0.025, momentum 0.92→0.99, WD=0.04
- Int6 per-row + zstd-22
- 3-seed: 1.1271, 1.1286, 1.1284 (mean 1.1280, std 0.0015)

---

## #5 — 11L Efficient Partial XSA + FA3 + SWA/120 (1.1307 BPB)
**Author:** unnir | **Date:** Mar 20 | **Params:** 26.8M | **Artifact:** 15.89MB | **Steps:** ~6,976 @ 86ms

### Key Innovations
- **Efficient Partial XSA**: same XSA mechanism but on last 3 layers only (8,9,10). GQA-aware reshape avoids `repeat_interleave`. ~2ms overhead.
- **FlashAttention 3**: direct `flash_attn_3_func` calls (Hopper-optimized)
- **SWA every 120 steps** (vs 50): 13 checkpoints averaged during warmdown

### Architecture
- 11L, 512d, GQA, 3× MLP, OrthoInit, SmearGate, BigramHash(2048)
- U-Net (5+6), NTK RoPE, logit softcap=30
- Muon: lr=0.025, momentum 0.92→0.99/1500, WD=0.04
- Int6 per-row + zstd-22
- Single seed (1337): 1.1307 BPB

---

## #6 — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) (1.1428 BPB)
**Author:** thwu1 | **Date:** Mar 20 | **Params:** 25.5M | **Artifact:** 15.97MB | **Steps:** ~7,100 @ 84ms

### Key Innovations
- **Mixed int5/int6 quantization**: int5 (clip=15, range [-16,15]) for MLP, int6 (clip=31) for attention. **Saves ~1.86MB** → funds 10th layer.
- **BigramHash(10240)**: 10240 buckets (vs 4096). XOR hash `(36313*curr) ^ (27191*prev) % 10239`. dim=128→512. **-0.0008 BPB** vs 8192 buckets.
- **SWA start_frac=0.4**: averages 24 checkpoints from last 40% of warmdown only (more converged). Every 50 steps.
- **Magnitude pruning**: zero smallest 3% of weights in large matrices before quantization. Improves zstd compression.
- **Orthogonal init** + muP-scaled projections (`1/sqrt(2*num_layers)`)

### Architecture
- 10 layers, 512d, GQA (8/4), 3× MLP (relu²), SmearGate
- U-Net (5+5), tied embeddings, logit softcap=30
- FP16 for tok_emb and last-layer c_k

### Training
- Muon: lr=0.02, momentum 0.92→0.99/1500, WD=0.04
- Adam: tied_embed_lr=0.03, scalar_lr=0.02, WD=0.04
- Batch: 786K, seq=2048, warmdown=3000, grad_clip=0.3
- zstd-22 compression
- 3-seed: 1.14271, 1.14298, 1.14260 (mean 1.14276, std 0.00016)

### Ablations
| Change | BPB | Delta |
|--------|-----|-------|
| 9L int6 baseline | 1.1485 | — |
| + int5 MLP + 10th layer | 1.1453 | -0.0032 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA start_frac=0.4 | 1.1446 | -0.0006 |
| + BigramHash 8192 | 1.1434 | -0.0012 |
| + BigramHash 10240 | 1.1426 | -0.0008 |

---

## #7 — Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA (1.1458 BPB)
**Author:** Raahil Shah | **Date:** Mar 20 | **Params:** 22.4M | **Artifact:** 15.81-15.89MB | **Steps:** ~7,380 @ 81.3ms

### Architecture
- **9 layers** (not 10/11), 512d, GQA (8/4), **3× MLP** (1536 hidden), relu²
- SmearGate (gate init=0), BigramHash(4096, dim=128), OrthoInit + muP
- U-Net (4+5), tied embeddings

### Training
- Muon: lr=0.02, momentum 0.92→0.99/1500, **WD=0.04** (decoupled)
- Adam: tied_embed_lr=0.03, scalar_lr=0.02, **WD=0.01**
- SWA: every 50 steps when scale<0.5, ~30 checkpoints averaged
- Batch: 786K, seq=2048, warmdown=3000, grad_clip=0.3

### Quantization
- Int6 per-row (MLP+attn), FP16 (tok_emb, blocks.8.attn.c_k), fp32 (controls)
- zstd-22. Quantization gap: ~0.016 BPB
- 3-seed: 1.14597, 1.14656, 1.14492 (mean 1.14582, std 0.00082)

---

## #8 — 11L MLP3x + Int6 QAT + zstd-22 + Sliding Window (1.1502 BPB)
**Author:** aruniyer | **Date:** Mar 20 | **Params:** 26.5M | **Artifact:** ~15.4MB | **Steps:** ~10,070 @ 59.6ms

### Key Innovations
- **Int6 QAT with STE**: fake-quantize weights to multiples of 4 during forward pass. `q = round(q_raw/4)*4`. Gradients flow through via STE. Eliminates quantization gap.
- **zstd-22**: first submission to use zstd over zlib. Saves ~1.5MB.
- **11 layers** with 3× MLP funded by int6+zstd savings
- **Sliding window eval** (stride=64): **-0.034 BPB** free improvement

### Training
- Muon: lr=0.025, momentum 0.92→0.99/1500
- Adam: tied_embed_lr=0.035, scalar_lr=0.025
- Decoupled WD=0.04 on both Muon and Adam
- FP16 tied embedding export
- Batch: 524K, seq=1024, warmdown=3000

### Results
- 3-seed sliding: 1.15055, 1.15021, 1.14970 (mean 1.15015, std 0.00043)
- t-statistic: 313.20 (p << 0.001)

---

## #9 — SmearGate + OrthoInit + Muon WD (1.1556 BPB)
**Author:** aquariouseworkman | **Date:** Mar 19 | **Params:** 22.4M | **Artifact:** 15.88MB | **Steps:** ~12,047 @ 49.8ms

### Key Innovations (early adopter of several techniques)
- **SmearGate** (original implementation): gate init=3.0 (sigmoid≈0.95). `output = g*x + (1-g)*x_prev`
- **Orthogonal init**: all 2D matrices ≥64×64. Synergizes with Muon (both operate in spectral space).
- **Muon WD=0.01** (decoupled): tighter weight distributions → better int6 quantization
- **Int6 STE QAT**: per-row fake quantization during training. Quantization gap: **~0.0001 BPB** (near zero).
- **BigramHash(4096)**: hash `(prev*92821 + cur) % 4096`, dim=128→512

### Architecture
- 9 layers, 512d, GQA (8/4), 3× MLP (relu²), U-Net (4+5)
- Tied embeddings, logit softcap=30, RoPE base=10000
- Batch: 524K, seq=1024, warmdown=3000
- Sliding eval stride=64, zstd-22

### Results
- Single seed (1337): sliding BPB=1.1556, standard BPB=1.1891
- Peak memory: 11,340 MiB

---

## #10 — 73.7M Ternary U-Net + FP8 + 8192 BPE (1.1570 BPB)
**Author:** Ciprian-Florin Ifrim | **Date:** Mar 24 | **Params:** 73.7M | **Artifact:** ~15.99MB | **Steps:** ~6,530 @ 91.8ms

### RADICALLY DIFFERENT APPROACH

### Key Innovations
- **Ternary quantization {-1,0,+1}**: 1.6 bits/param via base-3 encoding (5 trits/byte). 73.7M params in 16MB.
- **8192 BPE vocabulary**: 8× larger than baseline 1024. Largest single win: **-0.42 BPB** (fewer tokens/byte).
- **FP8 (e4m3) QAT**: non-ternary params (embeddings, projections) trained with FP8 STE. Native H100 support.
- **YaRN**: RoPE extension to 2048 context from 1024 training length. Smooth frequency interpolation.
- **NeoMuon**: only 3 Newton-Schulz steps (vs 5). Saves ~190 training steps worth of compute.
- **10 layers, 768 dim** (wider, not deeper). 4× MLP (3072 hidden).
- **Factored embedding**: EMBED_DIM=254 (not 768). Saves ~4MB on embedding table.
- **LZMA preset=9**: 39% better compression than int8+zlib on ternary data.
- **Poly5 softcap**: `logit_softcap=10` with polynomial softcap
- **Sliding eval stride=16** (more aggressive than stride=64)
- **Temperature scaling**: optimal T=0.90 on training tokens

### Architecture Details
- 10L 768d, GQA (8/4), 4× MLP (relu²), U-Net (5+5)
- Tied embeddings with factored dim=254
- Zero-fraction at convergence: 33.5% (exploited by LZMA)
- z_loss=1e-4 (logsumexp regularization)
- QK_GAIN=2.25

### Results
- 3-seed sliding: mean 1.1570 (std 0.0007)
- Compression: ternary base-3 + LZMA → 15.99MB

---

## #11 — 10L Int6 QAT + Zstd MLP2.6x (1.1586 BPB)
**Author:** yahya010 | **Date:** Mar 19

- 10 layers, 512d, MLP hidden=1344 (2.625×), int6 QAT + zstd-22
- Muon momentum=0.99, sliding eval stride=64
- Seq=2048, FP16 embeddings

---

## #12 — Mixed Quant Int6/Int8 + Sliding Window (1.1630 BPB)
**Author:** aquariouseworkman | **Date:** Mar 19

- 9 layers, 512d, **3× MLP** (first to use 3× expansion)
- Int6 on block weights, int8 on embeddings
- Sliding window eval stride=64
- Batch increased 393K→524K tokens/step
- Quantization gap: only 0.0015 BPB (STE QAT)

---

## #13 — Muon WD + 10 Layer (1.1748 BPB)
**Author:** notapplica | **Date:** Mar 19

- 10 layers, 512d, sliding window eval
- FP16 tied embedding, Muon WD=0.02
- Overtone spectral embedding init (SVD power-law spectrum)
- Phase-transition residual mixing with sigmoid-scheduled init
- Artifact: ~14.7MB (smallest yet at that time)

---

## #14 — Sliding Window Eval (1.1925 BPB)
**Author:** Matthew Li | **Date:** Mar 19

- **Pure evaluation innovation** — training identical to baseline
- Sliding window with stride=64: tokens scored with 960+ context
- Eval time: 16s → 70s (4.4× slower but **-0.032 BPB free**)
- First submission to break 1.20 BPB

---

## #15 — LoRA TTT (1.1928 BPB)
**Author:** samacqua | **Date:** Mar 19

- Per-document test-time training with rank-8 LoRA on Q, V, lm_head
- Adam lr=0.01, 256-token chunks, single gradient step per chunk
- Documents batched (batch=64), sorted by length, LoRA reset between docs
- Eval time: ~60s on 8×H100

### Ablations (critical insight: most gain from eval strategy, not LoRA)
| Condition | BPB | Delta |
|-----------|-----|-------|
| Baseline (cross-doc) | 1.2278 | — |
| + Doc-isolated | 1.2168 | -0.0110 |
| + Stride eval | 1.1941 | -0.0337 |
| + LoRA TTT | 1.1910 | -0.0368 |

---

## #16 — 4k Seq Length (1.2014 BPB)
**Author:** Spokane Way | **Date:** Mar 19

- Seq_len=4096 with reduced batch (393K vs 524K)
- Only 8,394 steps (71.5ms/step)
- Muon momentum=0.99, WD=0.01
- **Actually regressed** vs sliding window eval at seq=1024

---

## #17 — 2048 Seq Length (1.206 BPB)
**Author:** Spokane Way | **Date:** Mar 18

- Seq_len=2048 (2× baseline). Slower steps (71ms vs 43ms) but better per-token quality.
- 11,564 steps vs 13,780 baseline. Tuned LRs.

---

## #18 — Int6 Mixed Precision (1.2147 BPB)
**Author:** Nan Liu | **Date:** Mar 18

- 10 layers (first to add depth). Mixed int8/int6: layers 3-6 at int6, rest int8.
- Quantization penalty reduced from 0.0093 → 0.0018 BPB.

---

## #19 — FP16 Embed (1.2197 BPB)
**Author:** Renier Velazco | **Date:** Mar 18

- FP16 tied embedding (not quantized to int8). Embedding sensitive to quantization.
- MLP hidden reduced 1024→992 to fit. Warmdown 1200→3600.
- MATRIX_LR increased 0.04→0.06.

---

## #20 — Naive Baseline (1.2244 BPB)
**Author:** OpenAI | **Date:** Mar 18

- 9 layers, 512d, 8 heads / 4 KV heads, 2× MLP, relu²
- 1024 vocab (sp1024), seq=1024, tied embeddings
- Int8 per-row + zlib-9 compression
- Muon (matrices) + Adam (embeddings/scalars)
- The starting point for all submissions.

---

## Technique Inventory (What Has Been Implemented)

### Architecture
- [x] GQA (Grouped Query Attention)
- [x] RoPE / Partial RoPE / YaRN
- [x] U-Net skip connections
- [x] SmearGate (bigram blending)
- [x] BigramHash (4096, 8192, 10240 buckets)
- [x] XSA (Exclusive Self Attention)
- [x] Value Embeddings (per-layer, selective)
- [x] LN Scale Factor
- [x] Logit softcap (tanh, poly)
- [x] Factored embeddings (254-dim)
- [x] ReLU², LeakyReLU(0.5)²
- [x] 3× and 4× MLP expansion
- [x] 9, 10, 11 layer configurations
- [x] Tied embeddings

### Quantization & Compression
- [x] Int8 per-row (baseline)
- [x] Int6 per-row (MLP + attention)
- [x] Int5 per-row (MLP only)
- [x] Ternary {-1,0,+1} with base-3 encoding
- [x] FP8 (e4m3) QAT
- [x] QAT with STE (fake quantization during training)
- [x] Late QAT (triggered by LR threshold)
- [x] GPTQ-lite clip search (5 percentiles, min MSE)
- [x] Magnitude pruning (3% threshold)
- [x] zlib-9, zstd-22, LZMA-9 compression
- [x] FP16 embedding passthrough

### Optimization
- [x] Muon optimizer (Newton-Schulz 3-5 steps)
- [x] Parallel Muon (parameter banking, overlapped comm)
- [x] NeoMuon (3-step NS for ternary)
- [x] Adam / AdamW (per-group LRs)
- [x] Muon momentum warmup (0.85/0.92 → 0.95/0.99)
- [x] Decoupled weight decay (0.01-0.04)
- [x] Gradient clipping (norm=0.3)
- [x] Orthogonal initialization + muP scaling
- [x] EMA (decay=0.997)
- [x] SWA (every 50-120 steps)
- [x] Warmdown scheduling (1200-3500 iters)
- [x] Wallclock-based LR decay

### Evaluation
- [x] Sliding window eval (stride=16, 64)
- [x] Test-time training with LoRA (rank-8)
- [x] Legal score-first TTT (SGD, 3 epochs, all blocks)
- [x] Temperature scaling
- [x] FlashAttention 3

### Tokenizer
- [x] sp1024 (baseline, all top submissions)
- [x] sp8192 (ternary submission only)

---

## What Has NOT Been Tried (from OpenAI wishlist + our analysis)

- [ ] JEPA
- [ ] Text diffusion
- [ ] H-net tokenization
- [ ] Universal Transformer (4hr)
- [ ] Megakernels
- [ ] State-space models (Mamba)
- [ ] E2E TTT / super long context
- [ ] Learning adapters on random linear maps
- [ ] Custom entropy coding (rANS/ANS) replacing zstd
- [ ] Per-tensor optimal bit allocation (sensitivity-based)
- [ ] Knowledge distillation from larger teacher
- [ ] Weight sharing with width scaling (Shikhar started this)
- [ ] Progressive training (start shallow, grow)
- [ ] FP8 training (compute, not just storage)
- [ ] N-gram cache + model mixture (legality unclear)
- [ ] Mixture of Experts (confirmed negative at this scale)
- [ ] Byte-level tokenizer (confirmed negative for throughput)
