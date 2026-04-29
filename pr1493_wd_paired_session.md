# PR1493 wd_paired — Real Run Session, 2026-04-29

This document records the session where `wd_paired` was actually exercised end-to-end on the PR1493 stacking branch (commit `74dc702`). The previous session's "wd_paired" numbers turned out to be bogus; this run is the first one whose log carries the smoking-gun marker that confirms paired-head Muon was honored.

## Why this run was needed

The previous session reported `wd_paired q_ttt = 1.08009` and treated it as a real stack on top of `wd` (1.08029). Post-mortem in that session uncovered that:

- During FS turbulence around 06:54 UTC (shards 78 / 92 / 112 / 122 / 127 had ownership/permission flickers while being re-materialized), `train_pr1493.py` was rolled back to commit `5f35bb9` (the pre-stacking version: no `paired_head_muon_enabled`, no `fold_iha_mixes`, no 3D NS5).
- HEAD locally ended up at `ded7e22`, not `74dc702`.
- The post-06:54 logs (`pr1493_paired_s42.txt`, `pr1493_wd_paired_s42.txt`, `pr1493_wd_paired_iha_s42.txt`) had **zero hits on the smoking-gun string `tagged=22`**. The flag was a no-op.
- The "wd_paired 1.08009" was actually `wd alone + tuned TTT`, which explains why it sat only 0.00020 BPB from `wd` alone (= within noise).

The IHA run also failed in that session with `KeyError: 'blocks.0.attn.c_q.weight'` because the pre-stacking file lacked the `fold_iha_mixes` step.

## Pre-launch hardening

Before any run, we proved the file is the real stacking version and added a guard wrapper that aborts if it ever drifts.

### Verified state of `train_pr1493.py`

| field | value |
|---|---|
| HEAD | `74dc7028a06a0f52e2ce23a925ef24404e93ca1b` |
| md5 | `968e5ab744772b096a8f9b521656019d` |
| git blob | `1e4f7b4391f9a82b0ca7f735bbbb0db6eea8e8ad` |
| size | 57003 bytes |
| stacking symbols | 6 (paired_head_muon_enabled, fold_iha_mixes, paired_head_zeropower, ns5_3d, stacking, tagged=22) |
| diff vs HEAD | none |

The 6 symbols are the exact set added by commit `74dc702` over `5f35bb9`. The blob sha is the canonical evidence — it can only match if the file is byte-identical to what HEAD points at.

### `safe_launch.sh`

A wrapper that re-asserts all of the above immediately before exec'ing torchrun. If any check fails (HEAD, md5, git blob, symbol count, working-tree diff vs HEAD, backup md5), it aborts with a non-zero exit before any GPU work happens. Self-tested on a no-op `echo` before being used to launch the real run.

### Evidence the run was real (multi-vantage)

Verified during the live run, not just from the parent shell:

1. **All 8 worker PIDs** (read directly from `/proc/<pid>/environ`) carried `PAIRED_HEAD_MUON_ENABLED=1`, `WD_SCHEDULE_ENABLED=1`, `RUN_ID=pr1493_wd_paired_s42`, `SEED=42`, `TTT_LR=0.007`, `TTT_EPOCHS=5`, `QK_GAIN_INIT=5.25`.
2. **Training log** (`logs/pr1493_wd_paired_s42.txt`) carries the line:
   ```
   muon:paired-head NS enabled for q/k matrices tagged=22
   ```
   `tagged=22` = 11 transformer blocks × (`c_q` + `c_k`). The previous bogus runs had this string **zero** times.
3. **Hyperparameter dump** at log start contains `paired_head_muon_enabled: True` and `wd_schedule_enabled: True`.
4. **File integrity verified during the live run** (HEAD/md5/blob unchanged, no diff vs HEAD, mtime still `2026-04-29 09:25:53`). No FS-rollback this time.
5. **Post-run integrity verified** — same hashes, same mtime, all workers exited cleanly.

## What we ran

Local 8×H100 (single host), torch 2.9.1+cu128, flash-attn-3 cu128_torch291.

Dataset: `kevclark/parameter-golf` SP8192, 128 train shards (24 GB), 1 val shard. Downloaded fresh from HF via `data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128`. All 128 train shards landed at exactly 200,001,024 bytes except shard 127 which is the tail at 15,466,554 bytes.

```bash
RUN_ID=pr1493_wd_paired_s42 \
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1 \
./safe_launch.sh torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

## Two attempts (one crash, one success)

### Attempt 1 — crashed at serialize step

- Train completed normally (4589/20000, wallclock cap, 588099 ms). `tagged=22` and `wd_schedule_enabled: True` confirmed live.
- Pre-quant post-EMA `val_bpb = 1.08594791` (eval_time 7404 ms).
- GPTQ collected 67 Hessians in 12.8 s, quantized int6 + int8 emb successfully.
- Then died with `ModuleNotFoundError: No module named 'brotli'` on every rank, in `serialize()`'s call to `_compress(quant_raw, h.compressor)` (default `compressor='brotli'`).
- Logs preserved as `logs/pr1493_wd_paired_s42.txt.attempt1_brotli_crash` and `logs/pr1493_wd_paired_s42.stdout.attempt1_brotli_crash`.

`brotli` is included in the Modal image (`run_pr1493_modal.py`) but missing from local `requirements.txt`. Fix: `pip install brotli` (added to `requirements.txt` in this commit so future fresh clones don't trip).

### Attempt 2 — clean

- Identical command. `tagged=22` / `wd_schedule_enabled` confirmed live again on all 8 ranks.
- Train cap at 4596/20000 (vs attempt #1 at 4589 — 0.15 % drift from matmul nondeterminism, expected).
- Pre-quant post-EMA `val_bpb = 1.08609904` (vs attempt #1 at 1.08594791 — Δ 0.00015, below the 0.0002 BPB noise floor).
- GPTQ → brotli serialize → `Total submission size: 16,029,924 bytes`.
- Quantized eval, sliding-window eval, and TTT eval all completed.

## Final results (Attempt 2)

```
pre   = 1.08609904   (post-EMA, pre-GPTQ)
q     = 1.09891460   (post-GPTQ, no sliding, no TTT)
q_sw  = 1.08209420   (post-GPTQ, sliding window)
q_ttt = 1.07974373   (primary metric)
size  = 16,029,924 B (29,924 over the 16 MB hard cap)
stop  = 4596/20000   (wallclock cap = 588152 ms)
```

### Comparison vs all comparators

| run | pre | q | q_sw | q_ttt | Δq_ttt vs raw |
|---|---|---|---|---|---|
| PR1493 raw, seed 42 | 1.08757 | 1.10014 | 1.08329 | 1.08103 | — |
| tuned TTT (lr=.007, ep=5) | 1.08757 | 1.10014 | 1.08329 | 1.08079 | −0.00024 |
| `wd` alone | 1.08650 | 1.09951 | 1.08269 | 1.08029 | −0.00074 |
| bogus "wd_paired" (paired flag was no-op) | — | — | — | 1.08009 | −0.00094 |
| **wd_paired (this run, real)** | **1.08610** | **1.09891** | **1.08209** | **1.07974** | **−0.00129** |

### Stack attribution

- vs `wd` alone: −0.00055 BPB (above the 0.0002 noise threshold; small but real).
- vs the bogus `wd_paired` from the prior session: −0.00035 BPB. **This 0.00035 is the actual signal contributed by paired-head Muon firing**, which the bogus run missed.
- vs PR1493 raw: −0.00129 BPB total stack.

### Plan-doc prediction vs reality

`pr1493_stacking_plan.md` projected from the old banked paired-head Muon run:

> half transfer: 1.0793-ish
> full transfer: 1.0783-ish

We landed at 1.07974 — slightly worse than the half-transfer estimate. **Paired-head Muon transfers only partially to the PR1493+WD architecture.** The pre-quant gain (−0.00040) was small (the plan said "roughly unchanged"); the q_sw gain (−0.00060 vs `wd`) was real but smaller than the −0.002 the plan extrapolated from the old banked impl.

Quant gap (`q − pre`):
- baseline: 0.01257
- wd alone: 0.01301
- **wd_paired: 0.01281** — slightly tighter than `wd` alone, looser than baseline. Paired-head Muon does not aggressively close the quant gap on this architecture, contrary to the plan's hope.

## Learnings / gotchas (for future sessions)

1. **`tagged=22` is the smoking-gun signal for `PAIRED_HEAD_MUON_ENABLED`.** It only prints when `h.paired_head_muon_enabled` is true and the for-loop in `Optimizers.__init__` ran across all 11 blocks × (c_q + c_k). Always grep for it post-launch. Zero hits = the flag was a no-op, full stop, regardless of what env vars say.
2. **Verify env on the actual worker PIDs**, not just the launching shell. `tr '\0' '\n' < /proc/<pid>/environ` for every rank. If any rank is missing the flag, the run is mixed and meaningless.
3. **The integrity guard must check git blob sha, not just md5.** md5 collisions are unlikely but blob sha is what git itself uses, so it's free and authoritative.
4. **`brotli` is a hard dependency** (default `compressor='brotli'` in train_pr1493.py:8). It was missing from local `requirements.txt` (only baked into the Modal image). Added in this commit. `lzma` is stdlib; `zstandard` is also unimported by default.
5. **Always preserve crashed logs as `*.attempt1_<reason>`** before re-launching. Re-launching overwrites `logs/<run_id>.txt`, so without the rename you lose the partial evidence.
6. **The 0.0002 BPB noise threshold** (from prior multi-seed runs) is the right gate. Δ = 0.00055 vs `wd` alone is above it, so this is a real win. Anything below 0.0002 is single-seed noise and should be re-run on at least one more seed before being trusted.
7. **Wallclock cap step varies by ~10–20 steps run-to-run** due to matmul/cudnn nondeterminism even with fixed seed (4589 vs 4596 here). Pre-quant `val_bpb` drift was 0.00015 BPB between attempts 1 and 2 of the same config — this is the within-config noise floor on this hardware.
8. **The plan-doc's paired-head Muon predictions were extrapolated from the old banked implementation** (different architecture). Real transfer to PR1493 is partial (~30 % of the predicted q_ttt gain). For future stack predictions sourced from old banked numbers, halve the projected gain by default.

## Where we stand on the leaderboard bar

The acceptance margin is approximately 0.0017–0.0020 BPB. We have:

- −0.00129 vs raw PR1493 (q_ttt 1.08103 → 1.07974)

We're 0.0004–0.0007 BPB short. Candidates to close it:

- **`wd_paired_iha`** — fixed IHA fold path now testable end-to-end since brotli is installed (the prior IHA failure was a separate harness bug, but the brotli problem would have masked any progress past GPTQ anyway).
- **`wd_strong`** — `WD_SCHED_LOW_FACTOR=0.50 WD_SCHED_HIGH_FACTOR=1.75`, only worth trying if a stronger schedule is genuinely better than default WD; the plan said "only if default WD still looks good."
- **Re-run on a second seed** to confirm the −0.00055 vs `wd` gap isn't single-seed luck before stacking further.

Submission size for `wd_paired` is 16,029,924 B = 29,924 over the 16 MB cap. Code minification still required regardless of which metric we pick.

## Files committed in this session

- `safe_launch.sh` — guard wrapper for future runs.
- `pr1493_wd_paired_session.md` — this document.
- `pr1493_priority_results.md` — appended row 6 (wd_paired).
- `requirements.txt` — added `brotli`.
- `logs/pr1493_wd_paired_s42.txt` — primary log of the real run.
- `logs/pr1493_wd_paired_s42.stdout` — torchrun stdout for the real run, includes safe_launch line.
- `logs/pr1493_wd_paired_s42.txt.attempt1_brotli_crash` — primary log of attempt 1.
- `logs/pr1493_wd_paired_s42.stdout.attempt1_brotli_crash` — torchrun stdout for attempt 1.

`train_pr1493.py.bak.74dc702` is **not** committed — it's identical to HEAD's `train_pr1493.py` (md5 `968e5ab…`) and would be redundant with the git blob already in the repo. `safe_launch.sh` requires it, so on a fresh clone, recreate it with:

```bash
cp train_pr1493.py train_pr1493.py.bak.74dc702
```
