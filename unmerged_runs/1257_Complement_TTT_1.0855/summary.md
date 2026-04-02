# PR #1257: 11L Complement Training + TTT + No-JEPA

**BPB: 1.0855**
**Status:** OPEN (non-standard: 8xA100, 3600s training)
**Author:** BoxiYu (Bersekas Tully)
**Date:** 2026-04-02

## Key Techniques

1. **Complement Training** - Novel training technique
   - COMPLEMENT_ENABLED=1, COMPLEMENT_ALPHA=0.5
   - Details in train_sota1006.py

2. **TTT (Test-Time Training)** - 3 epochs, lr=0.0005
3. **No-JEPA** - Explicitly disabling JEPA improves results (JEPA_ENABLED=0)

## Results Summary

| Configuration | Time | Steps | val_bpb | Compliant |
|--------------|------|-------|---------|----------|
| TTT disabled | 2400s | 10477 | 1.1071 | Yes |
| TTT enabled | 2400s | 10582 | 1.1182 | Yes |
| TTT enabled | 3600s | 17020 | 1.0863 | No |
| **Complement + TTT** | **3600s** | **17050** | **1.0876** | **Yes** |
| **Complement + TTT + No-JEPA** | **3600s** | **-** | **1.0855** | **Yes** |

## What's Novel
- **Complement Training** with alpha=0.5 - appears to be a regularization/augmentation technique
- Finding that disabling JEPA improves results
- Runs on A100s (not H100s) with longer training time

## What's Borrowed
- Base architecture from train_sota1006.py (SOTA-derived)
- TTT framework from existing PRs
- LeakyReLU(0.5)^2 from standard stack

## Compatibility with SOTA Stack
- Complement Training could be added to any training pipeline
- TTT at lr=0.0005 (lower than standard 0.005) is a tuning detail
- JEPA disabling is relevant if your stack includes JEPA

## CRITICAL NOTES
- Trained on 8xA100 (not H100) with 3600s budget (not 600s)
- The 1.0855 BPB is with 6x more training time than standard
- Not clear if this would match on 8xH100 @ 600s
- The compliant config achieves 1.1071 (TTT disabled, 2400s) which is close to SOTA
