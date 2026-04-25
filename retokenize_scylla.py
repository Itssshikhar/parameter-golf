"""Retokenize SP8192 FineWeb shards to Scylla TokenMonster format.

Reads SP8192 .bin shards, decodes to text via SentencePiece, re-encodes via
TokenMonster (998-token Scylla vocab), writes new .bin shards in the same format.

Shard format: 1024-byte header (256 int32s: magic=20240520, version=1, num_tokens)
+ uint16 token array.

Usage:
    python retokenize_scylla.py [--workers N]
"""
import argparse
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tokenmonster

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256

SRC_DIR = Path("data/datasets/fineweb10B_sp8192")
DST_DIR = Path("data/datasets/fineweb10B_scylla")
SP_MODEL = "data/tokenizers/fineweb_8192_bpe.model"
TM_VOCAB = "data/tokenizers/scylla.vocab"

CHUNK_TOKENS = 50_000


def read_shard(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = struct.unpack(f"<{HEADER_INTS}i", f.read(HEADER_INTS * 4))
        assert header[0] == MAGIC, f"Bad magic: {header[0]}"
        num_tokens = header[2]
        tokens = np.frombuffer(f.read(num_tokens * 2), dtype=np.uint16).copy()
    return tokens


def write_shard(path: Path, tokens: np.ndarray):
    header = [0] * HEADER_INTS
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{HEADER_INTS}i", *header))
        f.write(tokens.astype(np.uint16).tobytes())


def retokenize_shard(args):
    src_path, dst_path, sp_model_path, tm_vocab_path = args
    if dst_path.exists():
        existing = read_shard(dst_path)
        if len(existing) > 0:
            print(f"  SKIP {dst_path.name} (already exists, {len(existing)} tokens)", flush=True)
            return dst_path.name, len(existing), 0.0

    t0 = time.time()
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab = tokenmonster.load(tm_vocab_path)

    src_tokens = read_shard(src_path)
    n = len(src_tokens)

    all_tm_tokens = []
    for start in range(0, n, CHUNK_TOKENS):
        chunk = src_tokens[start:start + CHUNK_TOKENS].tolist()
        text = sp.Decode(chunk)
        tm_tokens = vocab.tokenize(text)
        all_tm_tokens.append(tm_tokens)

    combined = np.concatenate(all_tm_tokens)
    assert combined.max() < 998, f"Token ID {combined.max()} >= 998"

    write_shard(dst_path, combined)
    elapsed = time.time() - t0
    print(f"  {dst_path.name}: {n} SP8192 -> {len(combined)} TM998 "
          f"(ratio {len(combined)/n:.3f}, {elapsed:.1f}s)", flush=True)
    return dst_path.name, len(combined), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--src-dir", type=str, default=str(SRC_DIR))
    parser.add_argument("--dst-dir", type=str, default=str(DST_DIR))
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("fineweb_*.bin"))
    if not src_files:
        print(f"ERROR: No .bin files in {src_dir}")
        sys.exit(1)

    print(f"Retokenizing {len(src_files)} shards: {src_dir} -> {dst_dir}")
    print(f"SP model: {SP_MODEL}, TM vocab: {TM_VOCAB}")
    print(f"Workers: {args.workers}")

    tasks = [
        (f, dst_dir / f.name, SP_MODEL, TM_VOCAB)
        for f in src_files
    ]

    t_start = time.time()
    total_tokens = 0

    if args.workers <= 1:
        for task in tasks:
            name, n_tok, _ = retokenize_shard(task)
            total_tokens += n_tok
    else:
        with mp.Pool(args.workers) as pool:
            for name, n_tok, _ in pool.imap_unordered(retokenize_shard, tasks):
                total_tokens += n_tok

    elapsed = time.time() - t_start
    print(f"\nDone: {total_tokens:,} total Scylla tokens in {elapsed:.0f}s")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
