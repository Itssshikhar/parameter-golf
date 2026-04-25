# Closing the gap — 5-day execution plan (2026-04-25 → 2026-04-30)

**Where we are:** 1.1117 BPB (3-seed mean) on SP1024 with SWA w=256 + bank QAT + QK-Gain 2.5 + PKO. Currently #10 on the leaderboard if Scylla is counted.

**Where the field is:**
- **0.9485** Scylla (PR #1184) — same arch as #1060, swapped to a 998-token TokenMonster vocab. Tokenizer engineering, not ML.
- **1.0810** SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT (PR #1493).
- **1.0856** SP8192 + GPTQ Embeddings + SDClip + Loop45×2 (PR #1394) — the cleanest pure-architecture ML record without TTT.

**Realistic 5-day target:** **≤ 1.06 BPB without TTT, ≤ 1.055 with TTT.** Beating 1.0810 is plausible but not safe. Beating 0.9485 requires Scylla-class tokenizer work which we are deferring as Tier-2 risk.

**Strategy:** front-load the tokenizer + quantization + compression cluster on Day 1–2 because that single batch buys ~−0.04 BPB. Architecture wins (MuonEq-R, depth recurrence, parallel residuals, QK-Gain) are Day 3–4. TTT is Day 5 only if everything else lands clean.

---

## Day-by-day overview

| Day | Theme | Net expected ΔBPB | Cumulative target |
|---|---|---|---|
| **0** (today) | Branch off, sanity check, kick off SP8192 retokenize in background | 0 | 1.1117 |
| **1** | Tokenizer swap to SP8192 + SDClip + Brotli-11 + WD 0.085 + MLP 4× | −0.030 to −0.045 | ~1.07–1.08 |
| **2** | MuonEq-R + QK-Gain 5.0 + 3-seed verification | −0.005 to −0.008 | ~1.065–1.075 |
| **3** | Depth recurrence on layers 4,5 + 3-seed verification | −0.003 to −0.005 | ~1.060–1.072 |
| **4** | Parallel residuals from layer 7 + 3-seed verification | −0.002 to −0.004 | ~1.058–1.068 |
| **5** | (a) Legal score-first Muon TTT *or* (b) seed reproduction + write PR | −0.003 to −0.004 (if TTT) | ~1.055–1.065 |

The order is set so each day's changes are independent enough to ablate. If Day 1 underperforms, we keep Day 0's stack as fallback. If Day 3 breaks training, we keep Day 2's stack as fallback.

---

## Tier 1 — Day 1: tokenizer + quantization + compression overhaul

This is **the single highest-leverage day**. PR #1218 alone (with SP4096 + WD 0.085 + MLP 4×) achieved 1.0979 from a 1.1194 baseline, a −0.022 BPB jump. PR #1394 added SP8192 and SDClip on top for another −0.012, landing at 1.0856.

### 1.A Retokenize FineWeb to SP8192

**Why this is the biggest lever:** BPB = `CE_nats / bytes_per_token / ln 2`. SP8192 packs ~30% more bytes per token than SP1024 in English text. Even at identical CE in nats, BPB drops proportionally.

**Implementation:**

```bash
# Use @clarkkev's pre-tokenized data (fastest path, ~5 min download)
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# OR regenerate from scratch (~60-90 min on a single H100)
cat > data/tokenizer_specs_8192.json << 'EOF'
[{"name": "sp_bpe_8192", "kind": "sentencepiece_bpe", "vocab_size": 8192, "tokenizer_train_docs": 5000000}]
EOF
python3 data/download_hf_docs_and_tokenize.py \
  --output-root data --tokenizer-config data/tokenizer_specs_8192.json --skip-byte
```

**Train script changes (`train_gpt_swa.py`):**

```python
# Around line 35-52, change defaults:
data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
```

**Artifact size impact:** embedding matrix grows from `(1024, 512) = 524K` floats to `(8192, 512) = 4.2M` floats. At int8 that's 4.2MB — eats ~26% of the 16MB budget. Mitigated by:
- int8 SDClip with k=20 on embeddings (already smaller than fp16, see 1.B)
- Brotli compression of int8 embeddings (typically 60-70% of raw bytes — see 1.C)
- Net: ~2.5MB, still leaves ~13MB for the rest

**Risk:** Embedding params dominate the model size. We may need to GPTQ-quantize the embedding matrix at int8, not int6, because embeddings are more sensitive — PR #1394 confirms this experimentally.

**Expected gain:** −0.025 to −0.035 BPB just from this swap.

**Fallback if SP8192 doesn't fit:** SP4096 instead. Same tooling, half the embedding size. Expected gain −0.015 to −0.020 BPB.

### 1.B SDClip — replace percentile search

**Why:** Our `quantize_int6_gptq` (lines 1313–1366 in `train_gpt_swa.py`) currently runs the GPTQ inner loop **5 times** with percentiles `[0.999, 0.9995, 0.9999, 0.99999, 1.0]` and picks the best by recon MSE. This is slow AND optimizes the wrong objective: recon MSE doesn't account for compressed size.

PR #1394 derives that for normally-distributed weights, `H(q) ≈ b - log₂(k) + const` where `c = k · σ`. Choosing `k` to land near 16MB is principled and one-shot.

**Implementation:** replace the percentile sweep in `quantize_int6_gptq`:

```python
def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128,
                      clip_sigmas=12.85):
    """Full GPTQ with SD-based clip: c = k * std(row)."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_sdclip(t32, clip_range, clip_sigmas)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    # SDClip: scale comes from row std, not max
    row_std = t32.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros_like(W, dtype=torch.int8)
    W_work = W.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        count = i2 - i1
        W1 = W_work[:, i1:i2].clone()
        Q1 = torch.zeros(rows, count, dtype=torch.int8)
        Err1 = torch.zeros(rows, count)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
            Q1[:, i] = q
            err = (w - q.float() * sf) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Q = Q[:, inv_perm]
    return Q, s
```

Add helper `_quantize_int6_sdclip(t32, clip_range, k)` for the no-Hessian fallback (e.g., 1D tensors).

**Embedding quantization:** add a separate path that uses `clip_range=127` (int8) and `clip_sigmas=20.0`:

```python
def quantize_int8_sdclip(weight, clip_sigmas=20.0):
    t32 = weight.float()
    row_std = t32.std(dim=1)
    s = (clip_sigmas * row_std / 127.0).clamp_min(1e-10).to(torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()[:, None]), -127, 127).to(torch.int8)
    return q, s
```

Wire into the embedding-quantization path at line ~395-420 of `train_gpt_swa.py`.

**Tuning knob:** if the artifact comes in too big, raise `clip_sigmas` (15, 20). If artifact is well under 16MB and quality might be compromised, lower it (10, 8). PR #1394 hand-tuned 12.85 to land near 16MB on SP8192 with 11L+512d.

**Expected gain:** −0.001 to −0.002 BPB plus ~3–5× faster GPTQ pass plus deterministic artifact size.

### 1.C Brotli-11 + byte-shuffle replaces LZMA-9

**Why:** PR #1089/#1179/#1218 all moved to Brotli. On int6 quantized tensors, byte-shuffle reorders bytes so high-order bits become contiguous, which Brotli compresses better than the interleaved layout. Reported ~5% smaller artifact than LZMA on identical inputs.

**Implementation:** replace the compression block at lines 2189–2252:

```python
import brotli
import struct

def byte_shuffle(arr_bytes: bytes, n_groups: int) -> bytes:
    """Shuffle bytes into n_groups streams.
    Useful when consecutive bytes have low entropy in different positions."""
    arr = bytearray(arr_bytes)
    n = len(arr)
    rem = n % n_groups
    if rem:
        arr.extend(b'\x00' * (n_groups - rem))
        n = len(arr)
    out = bytearray(n)
    stride = n // n_groups
    for g in range(n_groups):
        for i in range(stride):
            out[g * stride + i] = arr[i * n_groups + g]
    return bytes(out[:n - (n_groups - rem if rem else 0)])

def byte_unshuffle(arr_bytes: bytes, n_groups: int, original_n: int) -> bytes:
    arr = bytearray(arr_bytes)
    rem = original_n % n_groups
    n = original_n + ((n_groups - rem) if rem else 0)
    if len(arr) < n:
        arr.extend(b'\x00' * (n - len(arr)))
    out = bytearray(original_n)
    stride = n // n_groups
    for g in range(n_groups):
        for i in range(stride):
            idx = i * n_groups + g
            if idx < original_n:
                out[idx] = arr[g * stride + i]
    return bytes(out)

# At quantization time, replace lzma.compress(...) with:
quant_blob = brotli.compress(byte_shuffle(quant_raw, 4), quality=11)

# At reconstruction time:
quant_raw = byte_unshuffle(brotli.decompress(quant_blob_disk), 4, original_n)
```

The decoder needs to be embedded in the self-extracting wrapper. Add `import brotli` to the wrapper template and verify it's available in the Modal image (`pip install brotli`).

**Expected gain:** ~5% smaller artifact, no BPB change directly. The headroom this creates is what enables 1.D and 1.E.

### 1.D Bump WD to 0.085 (matrices) + 0.085 (embeddings)

**Why:** PR #1218's deepest insight: matrix RMS strongly correlates (R²≈0.99) with compressed size. Higher WD shrinks weight magnitudes → smaller compressed artifact for the same architecture.

**Train script changes:**

```python
# Lines 85-86:
muon_wd = float(os.environ.get("MUON_WD", 0.085))
adam_wd = float(os.environ.get("ADAM_WD", 0.02))  # scalars don't need much WD

# Add new knob (line ~90):
embed_wd = float(os.environ.get("EMBED_WD", 0.085))
```

Then in `Optimizers.__init__` or wherever the param groups are constructed, add the embedding param group with `weight_decay=embed_wd` (not `adam_wd`).

**Tuning:** PR #1285 went 0.085 → 0.090 for ~5% extra compression and **−0.0005 BPB**. We can defer that optimization until Day 2.

**Expected gain:** indirect through 1.E. Direct impact ~−0.001 BPB.

### 1.E MLP multiplier 3× → 4×

**Why:** PR #1218's removal of TTT, QAT, gated attention, value residuals, hash embeddings, smear gate — combined with the bigger MLP and higher WD — landed at 1.0979 vs 1.1194. Almost all of that delta is the bigger MLP + tokenizer.

**Train script changes:**

```python
# Line 57:
mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
```

That's it for code; the MLP class already takes `mlp_mult`.

**Artifact impact:** MLP grows from `dim * mlp_mult * 2 = 512 * 3 * 2 = 3072` params per MLP weight to `512 * 4 * 2 = 4096`. With 11 layers × 2 weights × 4096 elements = ~90K extra elements per MLP = ~0.9M total params, ~570KB at int6. Brotli + WD 0.085 should absorb this.

**Risk:** If artifact pops over 16MB, raise SDClip `k` to 14–15 first; if still over, drop to MLP 3.5× or keep BigramHash smaller.

**Expected gain:** −0.005 to −0.010 BPB (PR #1218 attributes ~half of its delta to bigger MLP, half to tokenizer).

### Day 1 verification

Single-seed sanity run, then 3-seed if it lands clean:

```bash
RUN_ID=day1_sp8192_sdclip_brotli SEED=1337 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  MUON_WD=0.085 ADAM_WD=0.02 EMBED_WD=0.085 \
  MLP_MULT=4.0 \
  COMPRESSOR=brotli \
  MATRIX_CLIP_SIGMAS=12.85 EMBED_CLIP_SIGMAS=20.0 \
  TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 \
  SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5 \
  PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 \
  torchrun --standalone --nproc_per_node=8 train_gpt_swa.py
```

**Pass criteria:**
- Artifact < 16,000,000 bytes
- Training < 600s
- Sliding BPB < 1.090 (single seed)
- No NaN/Inf at any logged step

**If pass:** launch 3-seed run via `run_swa_modal.py` updated to point at the new dataset. Expected 3-seed mean: 1.07–1.08.

**If artifact > 16MB:** raise `MATRIX_CLIP_SIGMAS=14.0`, then `EMBED_CLIP_SIGMAS=22.0`, then drop `MLP_MULT=3.5`, then fall back to SP4096.

**If sliding BPB > 1.10:** something broke in the pipeline. Suspect SDClip clip values too aggressive; lower `MATRIX_CLIP_SIGMAS` to 10. Or roll back tokenizer to SP1024 to isolate which change caused the regression.

---

## Tier 1 — Day 2: optimizer (MuonEq-R + QK-Gain)

### 2.A MuonEq-R: row-normalize gradients before Newton-Schulz

**Why:** Documented in PR #1217 (origin), used in PR #1285/#1334/#1394/#1493. Row-normalizing the gradient matrix before NS5 prevents row-magnitude correlations from biasing the orthogonalization. Reported ~−0.001 BPB at zero step-time cost.

**Implementation:** in our `Muon` optimizer (`train_gpt_swa.py:138`), find the section where `update = zeropower_via_newtonschulz5(update, steps=backend_steps)` is called (around lines 240–256). Add the row-normalize step right before:

```python
# In Muon.step, both the sharded and non-sharded paths
update = buf if not nesterov else g.add(buf, alpha=momentum)

# NEW: row-normalize before NS
if group.get("row_normalize", False):
    row_norms = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
    update = update / row_norms.to(update.dtype)

update = zeropower_via_newtonschulz5(update, steps=backend_steps)
```

**Param-group change:** in the optimizer construction, add `row_normalize=True` to the bank Muon param group:

```python
# wherever Muon is instantiated:
muon = Muon(bank_params, lr=matrix_lr, momentum=muon_momentum, ...,
            row_normalize=bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1"))))
```

**Risk:** none reported across 4+ records. Safe to enable by default.

**Expected gain:** −0.001 to −0.002 BPB.

### 2.B QK-Gain init 2.5 → 5.0

**Why:** PR #1493 reports monotonic improvement from 4.0 → 5.25 sweep. PR #1413 cites ~−0.003 BPB from 4.0 → 5.0. Our 2.5 was tuned for SP1024 + smaller MLP — very likely too conservative for the new stack.

**Implementation:** single env var:

```bash
QK_GAIN_INIT=5.0
```

That's the only change. Code already supports it (`train_gpt_swa.py:51`).

**Tuning sweep:** if Day 2's 3-seed run lands clean, do a quick single-seed sweep over `{4.0, 5.0, 5.25}` on Day 3 and pick the best. The optimum may shift with our specific stack.

**Risk:** initial training instability if too high — watch for NaN in first 100 steps. PR #1493 reports 5.25 trains stably end-to-end.

**Expected gain:** −0.002 to −0.003 BPB.

### Day 2 verification

```bash
RUN_ID=day2_muoneqr_qk5 SEED=1337 \
  ... (Day 1 env) \
  MUON_ROW_NORMALIZE=1 QK_GAIN_INIT=5.0 \
  torchrun ...
```

**Pass criteria:** 3-seed mean < Day 1 mean by ≥ 0.003 BPB.

**Cumulative target after Day 2:** 1.065–1.075.

---

## Tier 2 — Day 3: depth recurrence on layers 4,5

### 3.A Wire the encoder/decoder index lists

**Why:** PR #1285 (1.0912) and PR #1334 (1.0897) both add depth recurrence on layers 4,5 activated at 50% of training. PR #1493 extends to 3 layers. Reported ~−0.003 BPB. Zero extra parameters because weights are shared.

**Concept:** the model has 11 physical blocks. With recurrence, the schedule iterates through them as: encoder = `[0, 1, 2, 3, 4, 5, 4]`, decoder = `[5, 4, 5, 6, 7, 8, 9, 10]`. Layers 4 and 5 each get re-applied. The forward pass becomes a virtual 13-layer network using only 11 sets of weights. Activated at 50% of training so early steps don't pay the recurrence wallclock cost.

**Implementation in `GPT.__init__`:** add config knobs and build the index lists:

```python
# config (top of script):
recur_layers = os.environ.get("RECUR_LAYERS", "")  # e.g. "4,5"
recur_start_frac = float(os.environ.get("RECUR_START_FRAC", 0.5))

# in GPT.__init__, after self.blocks construction:
loop_layers = [int(x) for x in recur_layers.split(",") if x.strip()]
self.recur_active = False
if loop_layers:
    loop_start = min(loop_layers)
    loop_end = max(loop_layers)
    loop_seg = list(range(loop_start, loop_end + 1))
    all_indices = list(range(loop_start)) + loop_seg + loop_seg + list(range(loop_end + 1, num_layers))
    num_enc = len(all_indices) // 2
    self.encoder_indices_recur = all_indices[:num_enc]
    self.decoder_indices_recur = all_indices[num_enc:]
else:
    self.encoder_indices_recur = self.decoder_indices_recur = None

# baseline indices (no recurrence):
self.encoder_indices = list(range(num_encoder_layers))
self.decoder_indices = list(range(num_encoder_layers, num_layers))
```

**In `forward_logits`:** drive iteration by `*_indices_recur` when `self.recur_active`:

```python
enc_iter = (self.encoder_indices_recur if self.recur_active and self.encoder_indices_recur
            else self.encoder_indices)
dec_iter = (self.decoder_indices_recur if self.recur_active and self.decoder_indices_recur
            else self.decoder_indices)

for i in enc_iter:
    x = self.blocks[i](x, x0)
    skips.append(x)

for skip_idx, i in enumerate(dec_iter):
    if skip_idx < self.num_skip_weights and skips:
        ...
    x = self.blocks[i](x, x0)
```

**Activation hook:** in the training loop, set `model.recur_active = True` when `step >= recur_start_frac * total_steps`. With DDP, also set on `model.module.recur_active`.

**Skip-weight count:** `num_skip_weights = min(len(encoder_indices), len(decoder_indices))`. Recurrence makes both sides longer (e.g., 7 + 6 = 13 virtual layers split into 7/6); ensure the U-Net skip parameter count matches the longer schedule. May need to extend `self.skip_weights` to the recurrence-aware length.

**Expected gain:** −0.002 to −0.004 BPB.

**Risk:** activation timing matters. PR #1493 found that "always-on" recurrence loses to step-time penalty; delayed activation at 50% recovers most of the depth signal. Test single-seed first.

### 3.B Fewer training steps after recurrence kicks in

**Why:** After step `recur_start_frac * total_steps`, each step takes ~1.18× longer (2 extra block applications out of 11). Total wallclock budget is fixed at 600s. The model trades step count for virtual depth.

**Action:** none — the wallclock budget naturally shortens the post-activation portion. Just verify total step count stays sane in the log.

### Day 3 verification

```bash
RUN_ID=day3_depthrecur SEED=1337 \
  ... (Day 2 env) \
  RECUR_LAYERS=4,5 RECUR_START_FRAC=0.5 \
  torchrun ...
```

**Pass criteria:** 3-seed mean < Day 2 mean by ≥ 0.002 BPB. Sanity: log line confirming `recur_active=True` triggered around step `total/2`.

**Cumulative target after Day 3:** 1.060–1.072.

---

## Tier 2 — Day 4: parallel residuals from layer 7

### 4.A Wire the parallel residual

**Why:** PR #1204 (origin), PR #1412 (clean variant), PR #1477 (record). From layer 7 onward, attention and MLP read from the same input and add to the same residual stream:

```
# Sequential (layers 0-6, unchanged):
h = x + attn_scale * Attn(norm(x))
x_out = h + mlp_scale * MLP(norm(h))

# Parallel (layers 7-10):
x_out = x + attn_scale * Attn(norm(x)) + mlp_scale * MLP(norm(x))
```

Both reads from the same `x`. Removes attn→MLP serial dependency for those layers. PR #1412 attributes ~−0.002 BPB; PR #1477 confirms it stacks with TTT.

**Implementation in `Block.forward` (around `train_gpt_swa.py:828`):**

```python
parallel_start = int(os.environ.get("PARALLEL_START_LAYER", "7"))

# in Block.__init__, store layer_idx:
self.layer_idx = layer_idx

# in Block.forward:
if self.layer_idx >= self._parallel_start:
    # parallel: both read from attn_src (= x_in)
    attn_in = self.attn_norm(attn_src) * self.ln_scale_factor
    mlp_in = self.mlp_norm(attn_src) * self.ln_scale_factor
    attn_out, raw_v = self.attn(attn_in, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
    mlp_out = self.mlp(mlp_in, up_w, down_w)
    x_out = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out \
                  + self.mlp_scale.to(x_in.dtype)[None, None, :] * mlp_out
else:
    # sequential (existing code)
    attn_out, raw_v = self.attn(self.attn_norm(attn_src) * self.ln_scale_factor, ...)
    x_out = x_in + self.attn_scale[None, None, :] * attn_out
    x_out = x_out + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
return x_out, raw_v
```

Pass `parallel_start` from `GPT.__init__` to each block: `block._parallel_start = parallel_start`.

**Variant (optional, PR #1204):** separate residual lanes with a learned `lane_merge` scalar. We will start with the simpler PR #1412 version because it's lower risk. Upgrade only if there's headroom.

**Expected gain:** −0.001 to −0.003 BPB.

**Risk:** parallel residuals change the attention/MLP gradient flow. Should be stable, but watch the first 100 steps of training for divergence. If unstable, switch the layer threshold to 9 (only last 2 layers) and observe.

### Day 4 verification

```bash
RUN_ID=day4_parallel_resid SEED=1337 \
  ... (Day 3 env) \
  PARALLEL_START_LAYER=7 \
  torchrun ...
```

**Pass criteria:** 3-seed mean < Day 3 mean by ≥ 0.001 BPB.

**Cumulative target after Day 4:** 1.058–1.068.

---

## Tier 3 — Day 5: legal score-first Muon TTT (optional)

**Decision rule for Day 5:**
- If Days 1–4 land at ≤ 1.060: skip TTT, run 3-seed final reproduction, write the submission PR.
- If Days 1–4 land at 1.060–1.075: add TTT, expect another −0.003 to −0.004 BPB.
- If Days 1–4 land at > 1.075: do not add TTT. Spend Day 5 debugging the regression.

### 5.A Score-first protocol

**Compliance (Issue #1017):**
1. Causality — sliding window eval is causal.
2. Normalized distribution — standard softmax.
3. **Score before update** — every chunk fully scored under `torch.inference_mode()` before any gradient step.
4. Single pass — each token scored exactly once.

**Pseudocode:**

```python
def legal_ttt_eval(model, val_tokens, chunk_size=32768, ttt_lr=0.002, ttt_epochs=3,
                   ns_steps=3, h_high=2.1, h_low=1.75):
    chunks = chunkify(val_tokens, chunk_size)
    total_loss = 0.0
    total_bytes = 0
    ttt_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]

    for chunk_idx, chunk in enumerate(chunks):
        # PHASE 1: SCORE under inference_mode
        with torch.inference_mode():
            chunk_nll, chunk_bytes = sliding_window_score(model, chunk)
        total_loss += chunk_nll * len(chunk)
        total_bytes += chunk_bytes

        # Distributed sync of NLL for entropy gating
        cls_t = torch.tensor(chunk_nll, device=device, dtype=torch.float64)
        ctc_t = torch.tensor(len(chunk), device=device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(cls_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(ctc_t, op=dist.ReduceOp.SUM)
        global_chunk_nll = (cls_t / ctc_t).item()

        # Entropy-adaptive epoch count
        if global_chunk_nll > h_high:    epochs = 4
        elif global_chunk_nll < h_low:   epochs = 2
        else:                             epochs = 3

        # Skip update on last chunk (no future to adapt for)
        if chunk_idx == len(chunks) - 1: break

        # PHASE 2: TRAIN on already-scored chunk (Muon NS-3)
        cos_lr = ttt_lr * (0.5 * (1 + math.cos(math.pi * chunk_idx / len(chunks))))
        for _ in range(epochs):
            for x_batch, y_batch in chunk_batches(chunk):
                loss = model(x_batch, y_batch)
                loss.backward()
                # Muon-style update (no SGD)
                with torch.no_grad():
                    for p in ttt_params:
                        if p.grad is None: continue
                        g = p.grad.detach().float()
                        if g.ndim >= 2:
                            g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                        p.data.add_(g.to(p.dtype), alpha=-cos_lr)
                        p.grad = None

    return total_loss / total_bytes / math.log(2)
```

**Hyperparameters from PR #1148:**
- chunk_size = 32,768 tokens
- ttt_lr = 0.002 (cosine across chunks)
- ns_steps = 3 (5 exceeds 600s eval budget, 3 lands ~480s)
- h_high = 2.1 nats, h_low = 1.75 nats (entropy gating)

**Eval budget:** ~480s for TTT + ~80s for final sliding eval ≈ 560s. Inside the 600s eval cap.

**Expected gain:** −0.003 to −0.004 BPB. Strongly correlated with how much room there is in the model — if our pre-TTT BPB is already near the SOTA frontier, expect smaller gains.

**Risk:**
- **NCCL collective mismatch** if epoch counts diverge across ranks → watchdog timeout. PR #1148 explicitly fixed this with global NLL sync. Get this right or eval breaks.
- **Eval time overrun** if `ns_steps=5` accidentally used. Verify timing on first chunk.

### 5.B Skip TTT and reproduce instead

If Day 4 already landed at ≤ 1.060:
1. Run a clean 3-seed reproduction with the final config.
2. Verify all 3 seeds pass: artifact < 16MB, train < 600s, sliding eval < 100s.
3. Update `swa_experiment.md` with the final results.
4. Open the upstream PR following the format used by PR #1493.

---

## Quick reference: env-var stack at each day

```bash
# Day 0 (current, baseline)
TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5
PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000
QK_GAIN_INIT=2.5 MLP_MULT=3.0 MUON_WD=0.04 ADAM_WD=0.04
VOCAB_SIZE=1024 DATA_PATH=./data/datasets/fineweb10B_sp1024
# Result: 1.1117

# Day 1: + tokenizer + SDClip + Brotli + WD + MLP4
VOCAB_SIZE=8192 DATA_PATH=./data/datasets/fineweb10B_sp8192
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
MUON_WD=0.085 ADAM_WD=0.02 EMBED_WD=0.085
MLP_MULT=4.0 COMPRESSOR=brotli
MATRIX_CLIP_SIGMAS=12.85 EMBED_CLIP_SIGMAS=20.0
# Target: 1.07-1.08

# Day 2: + MuonEq-R + QK-Gain 5.0
MUON_ROW_NORMALIZE=1 QK_GAIN_INIT=5.0
# Target: 1.065-1.075

# Day 3: + Depth recurrence
RECUR_LAYERS=4,5 RECUR_START_FRAC=0.5
# Target: 1.060-1.072

# Day 4: + Parallel residuals
PARALLEL_START_LAYER=7
# Target: 1.058-1.068

# Day 5 (optional): + Muon TTT
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
TTT_MUON=1 TTT_NS_STEPS=3 TTT_ENTROPY_ADAPT=1
TTT_ENTROPY_HIGH=2.1 TTT_ENTROPY_LOW=1.75
# Target: 1.055-1.065
```

---

## Failure-recovery decisions

If any day's 3-seed run does not improve over the previous day's, do **not** revert and try the next day's change in isolation. Instead:

1. **Single-seed ablation** with the new change disabled, holding all else equal. Confirm the regression is due to the new change.
2. If confirmed: **try the smaller-risk variant** (e.g., MLP 3.5× instead of 4.0×, or PARALLEL_START_LAYER=9 instead of 7).
3. If still regressing: **skip that day's change**, freeze the previous day's stack as our final, and move on.

Do not stack changes whose individual contribution is unverified. The 3-seed std on these results is 0.0002–0.0008, so anything within ±0.002 is noise.

---

## What we are NOT doing

1. **Scylla/TokenMonster tokenizer.** Highest possible upside but requires:
   - 4–24 hours of TokenMonster vocab construction on FineWeb (offline)
   - Retokenization of all training data
   - Rewriting the tokenizer LUT path to load `.meta.npz` format
   - Risk that our hand-tuned tokenizer is worse than `english-1024-clean-v1` pruned
   
   With 5 days, this risk-reward is poor. Reconsider if Days 1–4 finish under budget.

2. **Hessian-aware SDClip (PR #1412 non-record).** Reported ~−0.001 BPB at λ=0.175; non-record status suggests author has low confidence. Skip.

3. **Hyperconnections (PR #1204).** Lane-merged residuals with learnable inter-lane writes. The simpler GPT-J variant (PR #1412) gives most of the gain. Skip the more complex version.

4. **Removing SWA from training.** None of the top records use SWA; we do. SWA at w=256 gave us seq_len=4096 affordably. Removing it reverts a working differentiator. Keep.

5. **Removing bank QAT.** Even though it didn't measurably move the needle in 3-seed data, it doesn't hurt and the code is in. Removing risks unknown interactions. Keep until/unless we need its bytes back.

---

## Final submission checklist (Day 5 evening)

- [ ] 3-seed mean BPB documented with std
- [ ] All 3 seeds: artifact < 16,000,000 bytes
- [ ] All 3 seeds: training wallclock < 600s
- [ ] All 3 seeds: eval wallclock < 600s (including TTT if used)
- [ ] `submission.json` populated with seed metadata
- [ ] `train_gpt.py` is the single self-contained file (no imports from `train_gpt_swa.py` etc.)
- [ ] Run command in README is copy-pasteable and reproduces seed 1337
- [ ] `requirements.txt` lists `brotli`, `flash_attn_3`, `torch==2.9.1`, etc.
- [ ] PR title follows leaderboard format: `Record: <techniques> — val_bpb <number> (3-seed mean)`
- [ ] PR body credits all upstream PRs whose techniques we adopted (#1060, #1218, #1394, #1493, #1285, #1204, #1148 at minimum)

---

## Tracking sheet

| Run | Stack | Pre-quant BPB | Sliding BPB | Artifact | Step avg | Steps | Status |
|---|---|---|---|---|---|---|---|
| 0 | SP1024 + bank QAT + SWA + PKO | — | 1.1117 (3-seed) | 15.97MB | 89ms | — | baseline ✅ |
| 1a | Day1 defaults: SP8192+SDClip+Brotli+WD+MLP4× SFL=5 seq=4096 | 1.1017 | 1.1171 | 15.73MB | 102ms | 5856 | BPB miss ❌ |
| 1b | Day1 + SFL=3 seq=2048 | 1.1070 | 1.1103 | 15.75MB | 93.6ms | 6409 | BPB miss ❌ |
| 1c | Day1 + SFL=3 seq=4096 | 1.0993 | 1.1136 | 15.71MB | 98ms | 6122 | BPB miss ❌ |
| 1d | Day1 + SFL=3 seq=4096 + MuonEq-R + QK5.0 | **1.0887** | **1.1016** | 16.59MB | 98ms | 6120 | **artifact FAIL** ⚠️ |
| 2 | + MATRIX_CLIP_SIGMAS=14.0 (pending) | TBD | TBD | | | | superseded |
| 3 | + Depth recurrence 4,5 | TBD | TBD | | | | superseded |
| 4 | + Parallel residuals @7 | TBD | TBD | | | | superseded |
| 5 | + Muon TTT | TBD | TBD | | | | optional |
| **S1** | **Scylla TM998 + MuonEq-R + QK5 + SFL=3 seq=4096** | **1.0681** | **1.0745** | 13.76MB | 92ms | 6491 | **new best** ✅ |
| **S2** | **Scylla raw 194 shards + SOTA config (no SWA, MLP3x, QK1.5, seq=2048)** | **1.0920** | **1.0841** | 11.59MB | 88.5ms | 6782 | **CE near-SOTA** ✅ |
| S3 | + Muon TTT | TBD | TBD | | | | optional |

**Key insight from 1d:** MuonEq-R + QK-Gain 5.0 gave the biggest single improvement (-0.011 pre-quant BPB) but MuonEq-R produces weight distributions that compress ~5% worse under Brotli. Need tighter SDClip (k=14.0) to fit 16MB.

**Quantization gap analysis:**
- Run 1a: gap = 0.015 (1.1017 → 1.1171)
- Run 1b: gap = 0.003 (1.1070 → 1.1103) — seq=2048 quantizes best
- Run 1c: gap = 0.014 (1.0993 → 1.1136)
- Run 1d: gap = 0.013 (1.0887 → 1.1016) — similar to 1c despite MuonEq-R
- Run S1: gap = 0.006 (1.0681 → 1.0745) — Scylla TM998 quantizes much better (tiny embedding, 2.24MB headroom)
- Run S2: gap = -0.008 (1.0920 → 1.0841) — sliding eval improves on non-sliding (expected with full attention)

**Scylla pivot (2026-04-25):** Runs 2-4 superseded by Scylla TokenMonster integration. The existing upstream Scylla PR (#1184) achieved 0.9485 BPB from 1.1122 (−0.1637) with zero architecture changes — just a 998-token TokenMonster vocab swap. We already have the vocab (`scylla.vocab`) and metadata (`scylla.meta.npz`). Retokenizing 128+1 SP8192 shards to TM998 format. The 998-token vocab saves ~3.7MB embedding budget (998×512 vs 8192×512), which resolves the 1d artifact size failure and frees headroom for MuonEq-R.

**Raw retokenization and bpt investigation (2026-04-25):**

Run S2 retokenized from raw `docs_selected.jsonl` (15.37M docs, 45GB) to get 194 train shards + 1 val shard, matching SOTA's data pipeline exactly. Used 32-worker multiprocessing, completed in ~8 minutes.

Key finding: **the BPB gap between us and SOTA is 100% from `bytes_per_token` (bpt), not from model quality.** Detailed analysis:

| Metric | S2 (ours) | SOTA (#1184) | Notes |
|--------|-----------|-------------|-------|
| CE sliding (nats) | **1.9327** | 1.9285 | Only 0.004 worse — model nearly as good |
| bpt (bytes/token) | **2.572** | **2.931** | 14% gap — THIS drives the BPB difference |
| BPB reported | 1.0841 | 0.9491 | BPB = CE / (bpt × ln2) |
| BPB if same bpt=2.93 | ~0.952 | 0.9491 | Would rank very close to #1 |

**Why bpt differs:** BPB is computed as `CE_nats / ln(2) × (token_count / byte_count)`, where `byte_count = Σ base_bytes_lut[token_id]` for all eval tokens. The `base_bytes_lut` is identical (verified: same md5 on vocab files, byte-identical meta.npz arrays). The val set text should be identical (same first 50K docs from same JSONL). Yet Scylla's val set averages 2.931 bytes per token while ours averages 2.572.

Possible causes for the bpt discrepancy:
1. **Different tokenmonster library version** — the tokenization algorithm produces different token sequences from the same text. We use v1.1.12 (latest); Scylla was submitted 2026-03-31 and may have used an earlier version with different ungreedy tokenization behavior.
2. **Different docs_selected.jsonl snapshot** — the upstream HF dataset may have been updated. Our sidecar says "not canonical 10B shard selection" (from a 50B shuffled train stream). If the canonical version has different first-50K docs, the bpt would differ.
3. **Scylla's retokenize.py is not available** — it's referenced in their README but not included in the submission record or git tree. We can't verify their exact pipeline.

**What we verified rules OUT:**
- Different tokenizer vocab (md5 identical: 54b30ead2cca047d3b85058144b47181)
- Different base_bytes LUT (byte-for-byte identical arrays in meta.npz)
- SP8192 round-trip artifacts (raw retokenization gives bpt=2.572, same as SP8192→TM998 path at 2.579)
- BOS prepending (adds only 50K tokens for 50K docs, negligible vs 913K token count gap)
- Different eval code logic (our byte counting matches SOTA's exactly)

**Implication:** Solving the bpt mystery is the single highest-leverage optimization. Closing the bpt gap from 2.57→2.93 would drop our BPB from 1.084→0.952 — a 0.132 improvement for zero model changes.

**Scylla run S1 config:**
```bash
DATA_PATH=./data/datasets/fineweb10B_scylla
TOKENIZER_PATH=./data/tokenizers/scylla.vocab
TOKENIZER_META_PATH=./data/tokenizers/scylla.meta.npz
VOCAB_SIZE=998
BIGRAM_VOCAB_SIZE=1024
BIGRAM_DIM=112
TRAIN_SEQ_LEN=4096
EVAL_SEQ_LEN=4096
SWA_WINDOW_SIZE=256
SWA_FULL_ATTN_LAYERS=3
QK_GAIN_INIT=5.0
PARTIAL_KEY_OFFSET=1
# MuonEq-R enabled via row_normalize=True in code
```

**Scylla run S2 config (raw 194 shards, SOTA-like):**
```bash
DATA_PATH=./data/datasets/fineweb10B_scylla_raw
TOKENIZER_PATH=./data/tokenizers/scylla.vocab
TOKENIZER_META_PATH=./data/tokenizers/scylla.meta.npz
VOCAB_SIZE=998
BIGRAM_VOCAB_SIZE=2816
BIGRAM_DIM=112
TRAIN_SEQ_LEN=2048
EVAL_SEQ_LEN=2048
SWA_WINDOW_SIZE=0  # full attention, no sliding window during training
MLP_MULT=3.0
QK_GAIN_INIT=1.5
XSA_LAST_N=11
# MuonEq-R enabled, EMA 0.997, SWA weight avg every 50 steps
# Uploaded: shikhar007/parameter-golf-gram-ns (run de841af9)
```

**Expected impact of Scylla swap:**
- BPB = CE_nats / bytes_per_token / ln2. Scylla bytes_per_token ≈ 4.13 (from meta), SP8192 ≈ 4.0. Similar. The win comes from smaller vocab (998 vs 8192): fewer classes to predict → lower CE. The upstream result was 0.9485 from 1.1122 = −0.1637 BPB.
- Embedding table: 998×512 = 511K params (int8 ≈ 499KB) vs 8192×512 = 4.2M (int8 ≈ 4.0MB). Saves ~3.5MB artifact budget.
- Logit projection: (batch, seq, 998) vs (batch, seq, 8192) — significantly cheaper fwd/bwd. Expect ~5-10ms/step savings.
- Token ratio: ~1.51x more tokens per byte → sequences cover less text. SWA w=256 covers ~690 bytes vs ~1024 bytes before.
