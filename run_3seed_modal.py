"""3-seed validation of best SWA config: seq4096, w=256, 5 full attn layers.

Runs seeds 1337, 42, 7 sequentially on 8xH100 to get a statistically
meaningful mean for comparison with current #1 (1.1147 BPB, 3-seed mean).
"""
import os
import modal

app = modal.App("parameter-golf-3seed")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    _hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(_hf_token_path):
        _hf_token = open(_hf_token_path).read().strip()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools")
    .run_commands(
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_gpt_swa.py", "/opt/train_gpt_swa.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=10800,  # 3 hours for 3 sequential runs
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def run_3seed():
    import subprocess
    import os
    import shutil

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")
    shutil.copy2("/opt/train_gpt_swa.py", "train_gpt_swa.py")

    # Dataset setup
    dataset_vol = "/data/fineweb10B_sp1024"
    dataset_local = "./data/datasets/fineweb10B_sp1024"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"

    if not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin"):
        print("Downloading dataset...", flush=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Itssshikhar/parameter-golf.git", "/tmp/repo"],
            check=True,
        )
        shutil.copytree("/tmp/repo/data", "./data", dirs_exist_ok=True)
        subprocess.run(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024"],
            check=True,
        )
        os.makedirs(dataset_vol, exist_ok=True)
        os.makedirs(tokenizer_vol, exist_ok=True)
        for f in os.listdir(dataset_local):
            shutil.copy2(f"{dataset_local}/{f}", f"{dataset_vol}/{f}")
        for f in os.listdir(tokenizer_local):
            shutil.copy2(f"{tokenizer_local}/{f}", f"{tokenizer_vol}/{f}")
        data_vol.commit()
        print("Dataset saved to volume.", flush=True)
    else:
        print("Dataset found in volume.", flush=True)
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local):
            shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    seeds = [1337, 42, 7]
    common_env = {
        "TRAIN_SEQ_LEN": "4096",
        "EVAL_SEQ_LEN": "4096",
        "SWA_WINDOW_SIZE": "256",
        "SWA_FULL_ATTN_LAYERS": "5",
        "BIGRAM_VOCAB_SIZE": "3072",
        "BIGRAM_DIM": "112",
        "WARMDOWN_ITERS": "4000",
    }

    all_results = []
    sliding_bpbs = []

    for seed in seeds:
        run_id = f"3seed_seq4096_swa256_full5_s{seed}"
        print(f"\n{'='*70}", flush=True)
        print(f"  SEED {seed} — {run_id}", flush=True)
        print(f"{'='*70}", flush=True)

        env = {
            **os.environ,
            **common_env,
            "SEED": str(seed),
            "RUN_ID": run_id,
        }

        # Clear torch compile cache between runs
        cache_dir = os.path.expanduser("~/.cache/torch_extensions")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        proc = subprocess.Popen(
            ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt_swa.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)
            lines.append(line)

        proc.wait()
        print(f"=== SEED {seed} DONE (exit code {proc.returncode}) ===", flush=True)

        # Extract sliding eval BPB from output
        for line in reversed(lines):
            if "final_int6_sliding_window" in line.lower() or "sliding" in line.lower():
                try:
                    # Try to extract the BPB number
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "sliding" in p.lower():
                            for candidate in parts[i:]:
                                try:
                                    val = float(candidate)
                                    if 1.0 < val < 1.3:
                                        sliding_bpbs.append(val)
                                        print(f"  >> Extracted sliding BPB: {val}", flush=True)
                                        break
                                except ValueError:
                                    continue
                            break
                except Exception:
                    pass
                if len(sliding_bpbs) == len(all_results) + 1:
                    break

        all_results.append(f"=== SEED {seed} ===\n" + "\n".join(lines))

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("3-SEED SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    for i, seed in enumerate(seeds):
        result = all_results[i]
        print(f"\n--- Seed {seed} ---", flush=True)
        for line in result.split("\n"):
            if any(k in line.lower() for k in [
                "step_avg", "post_ema", "final_int6_roundtrip",
                "final_int6_sliding_window", "sliding", "roundtrip",
                "pre_quant", "val_bpb",
            ]):
                print(f"  {line.strip()}", flush=True)

    if sliding_bpbs:
        mean_bpb = sum(sliding_bpbs) / len(sliding_bpbs)
        print(f"\n{'='*70}", flush=True)
        print(f"  Sliding BPBs: {sliding_bpbs}", flush=True)
        print(f"  MEAN SLIDING BPB: {mean_bpb:.4f}", flush=True)
        print(f"  Current #1:       1.1147", flush=True)
        print(f"  Delta:            {mean_bpb - 1.1147:+.4f}", flush=True)
        print(f"{'='*70}", flush=True)

    return "\n\n".join(all_results)


@app.local_entrypoint()
def main():
    log = run_3seed.remote()
    with open("3seed_run.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to 3seed_run.log")
