"""Retokenize FineWeb from raw docs_selected.jsonl using Scylla TokenMonster.

Matches the challenge pipeline: first 50K docs → val, rest → train,
shards of 100M tokens each. Uses multiprocessing for ~8-16x speedup.

Usage:
    python retokenize_scylla_raw.py [--workers 16] [--shard-size 100000000]
"""
import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8

DOCS_JSONL = Path("data/docs_selected.jsonl")
DST_DIR = Path("data/datasets/fineweb10B_scylla_raw")
TM_VOCAB = "data/tokenizers/scylla.vocab"

_vocab = None


def _init_worker(vocab_path):
    global _vocab
    import tokenmonster
    _vocab = tokenmonster.load(vocab_path)


def _tokenize_doc(text):
    encoded = _vocab.tokenize(text)
    if encoded is None or len(encoded) == 0:
        return np.array([], dtype=np.uint16)
    return encoded.astype(np.uint16)


def write_shard(path: Path, tokens: np.ndarray):
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default=str(DOCS_JSONL))
    parser.add_argument("--dst-dir", type=str, default=str(DST_DIR))
    parser.add_argument("--vocab", type=str, default=TM_VOCAB)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--num-val-docs", type=int, default=NUM_VAL_DOCS)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vocab: {args.vocab}")
    print(f"Docs: {args.docs}")
    print(f"Output: {dst_dir}")
    print(f"Shard size: {args.shard_size:,} tokens")
    print(f"Val docs: {args.num_val_docs:,}")
    print(f"Workers: {args.workers}")

    buf = np.empty((args.shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}
    doc_count = 0
    total_tokens = {"val": 0, "train": 0}

    def flush():
        nonlocal fill
        if fill == 0:
            return
        path = dst_dir / f"fineweb_{split}_{shards[split]:06d}.bin"
        write_shard(path, buf[:fill])
        print(f"  {path.name}: {fill:,} tokens", flush=True)
        shards[split] += 1
        fill = 0

    def add_tokens(toks):
        nonlocal fill
        pos = 0
        while pos < len(toks):
            take = min(args.shard_size - fill, len(toks) - pos)
            buf[fill:fill + take] = toks[pos:pos + take]
            fill += take
            pos += take
            if fill == args.shard_size:
                flush()

    def text_reader():
        with open(args.docs, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]

    t0 = time.time()

    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(args.vocab,))

    try:
        for toks in pool.imap(_tokenize_doc, text_reader(), chunksize=500):
            if doc_count == args.num_val_docs:
                flush()
                split = "train"

            if len(toks) > 0:
                total_tokens[split] += len(toks)
                add_tokens(toks)

            doc_count += 1

            if doc_count % 100_000 == 0:
                elapsed = time.time() - t0
                tps = sum(total_tokens.values()) / elapsed
                print(f"  docs: {doc_count:,}, tokens: {sum(total_tokens.values()):,}, "
                      f"time: {elapsed:.0f}s, {tps/1e6:.1f}M tok/s", flush=True)
    finally:
        pool.close()
        pool.join()

    flush()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Docs: {doc_count:,} (val: {min(doc_count, args.num_val_docs):,}, "
          f"train: {max(0, doc_count - args.num_val_docs):,})")
    print(f"Val tokens: {total_tokens['val']:,} in {shards['val']} shards")
    print(f"Train tokens: {total_tokens['train']:,} in {shards['train']} shards")
    print(f"Total tokens: {sum(total_tokens.values()):,}")
    print(f"Output: {dst_dir}")


if __name__ == "__main__":
    main()
