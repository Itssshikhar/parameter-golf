"""Continue raw retokenization from where it left off.

Skips val (done) and already-tokenized train documents,
then writes remaining train shards.
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
SHARD_SIZE = 10**8

DST_DIR = Path("data/datasets/fineweb10B_scylla_raw")
TM_VOCAB = "data/tokenizers/scylla.vocab"
DOCS_JSONL = Path("data/docs_selected.jsonl")

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

def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-val-docs", type=int, default=50_000)
    parser.add_argument("--skip-train-tokens", type=int, default=16_200_000_000)
    parser.add_argument("--start-shard", type=int, default=162)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dst_dir = DST_DIR
    buf = np.empty((SHARD_SIZE,), dtype=np.uint16)
    fill = 0
    shard_idx = args.start_shard
    train_tokens_seen = 0
    train_tokens_written = 0

    def flush():
        nonlocal fill, shard_idx
        if fill == 0:
            return
        path = dst_dir / f"fineweb_train_{shard_idx:06d}.bin"
        write_shard(path, buf[:fill])
        print(f"  {path.name}: {fill:,} tokens", flush=True)
        shard_idx += 1
        fill = 0

    def add_tokens(toks):
        nonlocal fill
        pos = 0
        while pos < len(toks):
            take = min(SHARD_SIZE - fill, len(toks) - pos)
            buf[fill:fill + take] = toks[pos:pos + take]
            fill += take
            pos += take
            if fill == SHARD_SIZE:
                flush()

    def text_reader():
        doc_count = 0
        with open(DOCS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                doc_count += 1
                if doc_count <= args.skip_val_docs:
                    continue
                yield json.loads(line)["text"]

    t0 = time.time()
    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(TM_VOCAB,))
    doc_count = args.skip_val_docs

    print(f"Skipping {args.skip_val_docs:,} val docs and {args.skip_train_tokens:,} train tokens")
    print(f"Starting from shard {args.start_shard}")

    try:
        for toks in pool.imap(_tokenize_doc, text_reader(), chunksize=500):
            doc_count += 1
            if len(toks) == 0:
                continue

            train_tokens_seen += len(toks)
            if train_tokens_seen <= args.skip_train_tokens:
                if doc_count % 500_000 == 0:
                    elapsed = time.time() - t0
                    print(f"  skipping: docs={doc_count:,}, train_seen={train_tokens_seen:,}, "
                          f"time={elapsed:.0f}s", flush=True)
                continue

            if train_tokens_seen - len(toks) < args.skip_train_tokens:
                skip_in_doc = args.skip_train_tokens - (train_tokens_seen - len(toks))
                toks = toks[skip_in_doc:]

            train_tokens_written += len(toks)
            add_tokens(toks)

            if doc_count % 100_000 == 0:
                elapsed = time.time() - t0
                print(f"  docs={doc_count:,}, written={train_tokens_written:,}, "
                      f"time={elapsed:.0f}s", flush=True)
    finally:
        pool.close()
        pool.join()

    flush()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Written {train_tokens_written:,} tokens in shards {args.start_shard}-{shard_idx-1}")

if __name__ == "__main__":
    main()
