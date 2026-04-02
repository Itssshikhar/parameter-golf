# PR #1271: Scylla Tokenizer Byte Accounting Audit

**Status:** OPEN (audit/investigation, not a submission)
**Author:** andrewbaggio1 (andreww)
**Date:** 2026-04-02

## Finding

**Sub-1.0 BPB claims from Scylla tokenizer (PR #1184) were a measurement error.**

With corrected `candidate.meta.npz`, the actual BPB is **1.1289, not 0.9485**.

## The Bug

PR #1184's `candidate.meta.npz` has 27 byte-fallback tokens (IDs 75-101) with
`base_bytes=3` instead of 1. These tokens represent single raw bytes but are counted
as 3 bytes each. This overcounts the byte denominator in the BPB formula, making the
score look ~4% better than it actually is.

## Decomposition of the BPB Gap

| Factor | BPB impact |
|--------|:---------:|
| Model quality (NLL difference) | +0.010 |
| **Byte accounting difference** | **+0.133** |
| Val text/token boundary differences | +0.037 |
| **Total** | **+0.180** |

**93% of the gap is byte accounting, not model quality.**

## Impact on Other Scylla PRs

This casts doubt on any Scylla-tokenizer submission that uses the original
`candidate.meta.npz`. Specifically:
- PR #1184: 0.9485 -> likely ~1.13 with correct accounting
- PR #1242 (closed): 1.0903 BPB - may also be affected
- PR #1274: 1.0876 BPB - needs verification

## What This Means for Us

- Always verify byte accounting when using non-standard tokenizers
- The Scylla tokenizer itself provides no meaningful BPB advantage over SP-1024
- The corrected Scylla stack lands at ~1.13 BPB, essentially same as SP-1024 at ~1.11-1.12
