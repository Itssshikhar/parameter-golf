# Parameter Golf — Baseline train_gpt.py Code Analysis

## Overview

The baseline `train_gpt.py` is a 1127-line self-contained GPT training script. It trains a small transformer on FineWeb data (8B tokens), quantizes the model to fit in 16MB, and evaluates via bits-per-byte (BPB). The challenge: lowest BPB within 16MB artifact + 10-min training on 8×H100s.

**Current SOTA: 1.1428 BPB | Baseline: 1.2244 BPB**

---

## Section 1: Config (Lines 1-95)

- **Lines 1-5**: Docstring — starting point, not a competition winner. Hard cap: 1500 lines.
- **Lines 7-28**: Imports. Notable: `zlib` (compression), `sentencepiece` (tokenizer), `DDP` (multi-GPU).
- **Lines 39-87**: Every hyperparameter as env var override. Key groups:
  - **Data**: Where shards and tokenizer live
  - **Training length**: 20K iterations, 524K tokens/step, 1024 seq length, 10-min wallclock cap
  - **Model shape**: 9 layers, 512 dim, 8 heads (4 KV heads = GQA), 2× MLP, 1024 vocab, tied embeddings
  - **Optimizer**: Different LRs for different param types (embeddings=0.6, matrices=0.04, head=0.008). Muon momentum starts at 0.85, warms to 0.95

---

## Section 2: Muon Optimizer (Lines 96-168)

### `zeropower_via_newtonschulz5` (96-109) — Core Trick

Orthogonalizes gradient matrices — makes all singular values equal to 1. The update treats every direction equally, instead of being dominated by the biggest gradient direction.

```
1. Normalize G to roughly unit norm
2. If tall matrix, transpose (algorithm prefers wide)
3. Repeat 5 times:
   A = X @ X.T          ← "how far from orthogonal am I?"
   B = -4.775*A + 2.032*A²   ← correction term
   X = 3.445*X + B @ X       ← refine toward orthogonal
4. Result: same "shape" as original gradient, but all singular values = 1
```

Magic numbers (3.4445, -4.7750, 2.0315) are polynomial coefficients for fast convergence in ~5 iterations. Much faster than SVD.

### `Muon.step()` (119-168) — Distributed Work-Sharding

- 8 GPUs all have same gradients (DDP synced them)
- Newton-Schulz is expensive, so GPU 0 orthogonalizes params 0,8,16..., GPU 1 does 1,9,17..., etc.
- Each GPU writes results into flat buffer (zeros elsewhere)
- One `all_reduce(SUM)` combines them — every GPU gets full set
- Each GPU applies update to local weights

Per parameter:
1. **Momentum**: `buf = 0.95 * buf + gradient`
2. **Nesterov**: `g = gradient + 0.95 * buf` (peek ahead)
3. **Orthogonalize**: `g = newton_schulz(g)`
4. **Scale**: multiply by `sqrt(rows/cols)` if tall
5. **Update**: `weight -= lr * g`

---

## Section 3: BPB Evaluation (Lines 174-278)

### `build_sentencepiece_luts` (180-204) — 3 Lookup Tables

For each token in vocab:
- **base_bytes**: UTF-8 byte count (e.g., "hello" = 5 bytes)
- **has_leading_space**: Starts with ▁ (SentencePiece space marker)?
- **is_boundary_token**: Special/control token?

Byte counting per predicted token:
```
bytes = base_bytes[token]
if has_leading_space[token] AND previous_token is NOT boundary:
    bytes += 1    ← count the space SentencePiece hid in ▁
```

### `eval_val` (219-278) — THE Metric

1. Split validation tokens across GPUs (each rank gets disjoint chunk)
2. For each batch of non-overlapping 1024-token sequences:
   - x = tokens[:-1], y = tokens[1:]
   - Forward pass → cross-entropy loss
   - Count bytes per predicted token using LUTs
3. All-reduce across GPUs
4. Final computation:
   ```
   val_loss = total_loss / total_tokens        (nats per token)
   bits_per_token = val_loss / ln(2)           (convert to bits)
   tokens_per_byte = total_tokens / total_bytes (tokenizer compression ratio)
   BPB = bits_per_token × tokens_per_byte      (THE number)
   ```

**Key insight**: BPB has two sides. Lower bits_per_token (better predictions) helps. Lower tokens_per_byte (better tokenizer) also helps. Tokenizer choice matters.

---

## Section 4: Quantization & Compression (Lines 288-422)

### Config (288-308)
- `CONTROL_TENSOR_NAME_PATTERNS` — attn_scale, mlp_scale, q_gain kept in fp32
- `INT8_KEEP_FLOAT_MAX_NUMEL = 65,536` — Small tensors stored as fp16 directly
- `INT8_CLIP_PERCENTILE = 99.99984%` — Clip outliers before quantizing

### `quantize_float_tensor` (321-340) — Core Math

**Per-row (2D matrices)**:
```python
clip_abs = 99.99984th percentile of |weights| per row
scale = clip_abs / 127
quantized = round(clipped_weights / scale)  # int8 values
```

**Per-tensor (vectors)**:
```python
scale = max(|tensor|) / 127
quantized = round(tensor / scale)
```

### `quantize_state_dict_int8` (342-399) — Treatment Per Tensor

```
NOT floating point → passthrough
Small (≤ 65,536 elements) → fp16 (or fp32 for controls)
Large float tensor → int8 (per-row for 2D, per-tensor for 1D)
```

### `dequantize_state_dict_int8` (401-422) — Reverse

```python
reconstructed = quantized.float() * scale
restored = reconstructed.to(original_dtype)
```

Difference = quantization error → directly hurts BPB.

---

## Section 5: Data Loading (Lines 425-494)

### `load_data_shard` (429-443)
Binary .bin format: [1024-byte header] [uint16 tokens...]
- Header[0] = 20240520 (magic), Header[1] = 1 (version), Header[2] = num_tokens

### `TokenStream` (446-474)
Sequential shard reader. Wraps around forever. Loads one shard at a time.

### `DistributedTokenLoader` (477-494)
Each GPU gets disjoint slice. Creates (x,y) pairs by shifting by 1.
With 8 GPUs + 524K tokens/step → 65K tokens per GPU per step.

---

## Section 6: Core Building Blocks (Lines 496-553)

- **RMSNorm (500-506)**: Normalize to unit RMS. Zero params. Faster than LayerNorm.
- **CastedLinear (509-513)**: Weights in fp32 (optimizer precision), cast to bf16 at forward time (speed).
- **Rotary (524-552)**: RoPE position embeddings. Pre-computes cos/sin tables. Gives relative position awareness.
- **apply_rotary_emb (549-552)**: 2D rotation per dimension-pair.

---

## Section 7: Model Architecture (Lines 555-724)

### CausalSelfAttention (555-603)

8 query heads, 4 KV heads (GQA). Head dim = 64.
- Q: 512→512, K: 512→256, V: 512→256 (GQA saves ~40% params)
- `q_gain`: learnable scalar per head (init 1.5, makes attention sharper)

Forward:
```
x → project Q/K/V → reshape to heads
→ RMSNorm Q,K → RoPE → scale Q by q_gain
→ FlashAttention (causal) → project back
```

### MLP (606-617)

```
x → fc: 512→1024 (expand) → ReLU → square → proj: 1024→512
```
ReLU² = ReLU followed by squaring. `proj._zero_init = True` — starts contributing nothing.

### Block (620-645)

One transformer layer with 3 tricks:
1. **Residual mixing**: `x = mix[0]*x + mix[1]*x0` (blend with original embedding)
2. **Scaled residuals**: `x = x + attn_scale * attn_output` (per-dim learned scaling)
3. **Pre-norm**: Normalize BEFORE attention/MLP

### GPT (648-724)

U-Net architecture: 4 encoder + 5 decoder blocks.
- Encoder saves intermediate outputs
- Decoder uses them as skip connections (reverse order)
- Tied embeddings: input table = output projection
- Logit softcap: `30 * tanh(logits/30)` → clamps to [-30, 30]
- Loss: cross_entropy(logits, targets)

---

## Section 8: Training Setup (Lines 727-921)

### Distributed + CUDA (742-769)
- `grad_accum_steps = 8 // world_size` — 1 micro-step per GPU with 8 GPUs
- FlashAttention forced, TF32 enabled

### Model Creation (826-844)
```python
base_model = GPT(...).to(device).bfloat16()
CastedLinear modules → float()           # weights in fp32
compiled_model = torch.compile(...)       # JIT compile
model = DDP(compiled_model)               # multi-GPU wrapper
```

### Optimizer Split (846-893)

| Optimizer | Parameters | LR |
|-----------|-----------|-----|
| Adam | Token embedding | 0.05 (tied) |
| Muon | 2D weight matrices | 0.04 |
| Adam | Control scalars (scales, mix, gain, skip) | 0.04 |
| Adam | lm_head (if untied) | 0.008 |

---

## Section 9: Training Loop (Lines 922-1060)

### LR Schedule (924-933)
Flat at 1.0 until warmdown, then linear decay to 0.0. Wallclock-adaptive.

### Compilation Warmup (937-961)
1. Save model + optimizer state
2. Run 20 steps (prime torch.compile)
3. Restore original state (throw away training)
4. Reset data loader

### Main Loop (967-1055)
Each iteration:
```
1. Validate if needed (pause timer)
2. Compute LR multiplier
3. Zero gradients
4. For each micro-step: forward → backward (accumulate gradients)
5. Update Muon momentum (0.85→0.95 over 500 steps)
6. Scale LRs by warmdown multiplier
7. Gradient clipping (if enabled)
8. All optimizers step
9. Check wallclock cap → all GPUs stop together
```

---

## Section 10: Serialization & Final Eval (Lines 1062-1127)

The endgame:
```python
# 1. Save raw model (~98MB)
# 2. Quantize to int8 (per-row + scales)
# 3. Compress with zlib level 9
# 4. Write artifact (~15MB)
# 5. CRITICAL: Roundtrip validation
#    Load artifact → decompress → dequantize → eval BPB
#    THIS is the final score
```

Roundtrip BPB = the competition metric. Quantization error directly hurts score.

---

## Parameter Count (Baseline 9L, 512d, 2× MLP, tied)

| Component | Params |
|-----------|--------|
| Token embedding (1024×512) | 524K |
| Attention per layer (Q+K+V+proj) | ~787K |
| MLP per layer (fc+proj) | ~1M |
| Control per layer (scales, mix, gain) | ~2K |
| Skip weights | ~2K |
| **Total (~9 layers)** | **~17.1M** |

---

## SOTA Techniques (1.1428 BPB)

| Technique | Individual Impact |
|-----------|------------------|
| Sliding window eval (stride=64) | -0.034 BPB |
| 3× MLP expansion | -0.029 BPB |
| 10-11 layers (funded by int5/6) | -0.013 to -0.024 BPB |
| Mixed int5 MLP + int6 attention | Saves ~1.86MB |
| zstd-22 compression | Saves ~1.5MB vs zlib |
| SmearGate + BigramHash | -0.012 each |
| SWA (last 40% of training) | -0.006 BPB |
| Lower LR (0.02) | -0.006 BPB |
| Weight decay 0.04 | -0.004 BPB |
| Orthogonal init | -0.006 BPB |
| Muon momentum 0.99 + warmup | -0.008 BPB |

---

## Key Constraints

- **16MB** = code bytes + compressed model bytes (decimal, not binary)
- **10 min** training on 8×H100 SXM
- **10 min** additional for evaluation
- **BPB** = bits_per_token × tokens_per_byte (tokenizer-agnostic)
- **Tokenizer CAN be changed** (more scrutiny on submissions)
- **Any package allowed** (library bytes don't count toward 16MB)
- **Beat SOTA by ≥0.005 nats, p<0.01, 3+ seeds**

---

## Untried Directions

- Novel tokenizer (larger vocab, byte-level)
- Depth recurrence / weight sharing (Shikhar started this)
- Mixture of Experts
- int4 quantization on least-sensitive layers
- Vector/codebook quantization
- Knowledge distillation
- Test-time training with LoRA during eval
- Sparse attention patterns
- Post-quantization fine-tuning
- Custom CUDA kernels for throughput
- JEPA / non-autoregressive architectures
- State-space models (Mamba-style)
- Hybrid architectures
