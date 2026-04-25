# Day 1 stack — new defaults baked into `train_gpt_swa.py`

All five Day 1 changes from `closing_the_gap_5day_plan.md` are now the **defaults** in `train_gpt_swa.py`. No env vars need to be set to get the Day 1 stack — just run `torchrun ... train_gpt_swa.py` and you get SP8192 + SDClip + Brotli + WD 0.085 + MLP 4×.

## Changed defaults

| Knob | Env var | Day 0 (was) | Day 1 (now) | Section |
|---|---|---|---|---|
| Dataset path | `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | `./data/datasets/fineweb10B_sp8192` | 1.A |
| Tokenizer path | `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | `./data/tokenizers/fineweb_8192_bpe.model` | 1.A |
| Vocab size | `VOCAB_SIZE` | `1024` | `8192` | 1.A |
| MLP multiplier | `MLP_MULT` | `3.0` | `4.0` | 1.E |
| Muon weight decay | `MUON_WD` | `0.04` | `0.085` | 1.D |
| AdamW (scalars) WD | `ADAM_WD` | `0.04` | `0.02` | 1.D |
| Embedding WD | `EMBED_WD` | (n/a) | `0.085` | 1.D |
| Matrix SDClip k | `MATRIX_CLIP_SIGMAS` | (n/a, percentile sweep) | `12.85` | 1.B |
| Embedding SDClip k | `EMBED_CLIP_SIGMAS` | (n/a, percentile=99.99984) | `20.0` | 1.B |
| Compressor | (hard-coded) | LZMA preset 9 | Brotli q11 + byte-shuffle (n_groups=4) | 1.C |

## Code-level changes (not env-controlled)

- **`quantize_int6_gptq`**: dropped 5-percentile sweep `[0.999, 0.9995, 0.9999, 0.99999, 1.0]` → single SDClip pass with `c = k · σ_row`. ~3–5× faster GPTQ.
- **`quantize_int6_per_row`** / **`_quantize_int6_sdclip`** (renamed from `_quantize_int6_percentile`): same SDClip rule.
- **`quantize_float_tensor`**: int8 path now uses SDClip k=20 instead of percentile=99.99984. Affects all embeddings + non-int6-cat tensors via `mixed_quantize_int6`.
- **Brotli compression**: 3 call-site swaps inside `main()`:
  - prune-size estimator
  - final compress to `final_model.int6.ptz`
  - eval round-trip decompress
- New helpers near top of file: `byte_shuffle`, `byte_unshuffle`, `brotli_byteshuffle_compress`, `brotli_byteshuffle_decompress` (8-byte big-endian size header prefix).

## Prerequisites before running

1. **SP8192 data on disk.** Either pull pre-tokenized:
   ```bash
   rm -f data/manifest.json
   MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
     python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
   ```
   …or regenerate from scratch (60–90 min on a single H100; see plan §1.A).
2. **`brotli` in the runtime image.** Add `pip install brotli` to the Modal image build. The script imports `brotli` at module load and will hard-fail otherwise.

## Run command

Single-seed sanity:

```bash
RUN_ID=day1_sp8192_sdclip_brotli SEED=1337 \
  TRAIN_SEQ_LEN=4096 EVAL_SEQ_LEN=4096 \
  SWA_WINDOW_SIZE=256 SWA_FULL_ATTN_LAYERS=5 \
  PARTIAL_KEY_OFFSET=1 BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 \
  torchrun --standalone --nproc_per_node=8 train_gpt_swa.py
```

## Pass criteria

- Artifact < 16,000,000 bytes
- Training wallclock < 600s
- Sliding BPB < 1.090 (single seed)
- No NaN/Inf at any logged step

## Recovery overrides

| Symptom | Override (in order) |
|---|---|
| Artifact > 16 MB | `MATRIX_CLIP_SIGMAS=14.0` → `EMBED_CLIP_SIGMAS=22.0` → `MLP_MULT=3.5` → fall back to SP4096 |
| Sliding BPB > 1.10 | `MATRIX_CLIP_SIGMAS=10.0` (clip too loose) → roll back to SP1024 to isolate |
| NaN early in training | `QK_GAIN_INIT=2.5` (no change yet — Day 2's QK 5.0 is not in defaults) |
| Brotli import error | Add `pip install brotli` to image |

## What's NOT in Day 1 defaults yet

These are Day 2+ changes still to come:
- MuonEq-R row-normalize (Day 2)
- QK-Gain init 5.0 (Day 2 — still 2.5 in defaults)
- Depth recurrence layers 4,5 (Day 3)
- Parallel residuals from layer 7 (Day 4)
- Legal score-first Muon TTT (Day 5)
