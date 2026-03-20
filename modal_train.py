"""
Modal launcher for parameter-golf maximalist batch training on 8xH100.

Usage:
    modal run modal_train.py

Requires Modal token configured:
    modal token set --token-id <ID> --token-secret <SECRET> --profile=exafunction
    modal profile activate exafunction
"""
import modal
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(
    REPO_ROOT, "records", "track_10min_16mb",
    "2026-03-20_MaximalistBatch", "train_gpt.py",
)

app = modal.App("parameter-golf-maximalist")

# Persistent volume for caching dataset across runs
vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "sentencepiece",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    # Add training script into the image
    .add_local_file(TRAIN_SCRIPT, remote_path="/workspace/train_gpt.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 45,  # 45 min total (data download + torch.compile + 10 min train + eval)
    volumes={"/data": vol},
    memory=131072,  # 128 GB RAM
)
def train():
    import subprocess
    import sys
    import time
    import shutil
    import json

    DATA_ROOT = "/data"
    DATASET_DIR = f"{DATA_ROOT}/datasets/fineweb10B_sp1024"
    TOKENIZER_DIR = f"{DATA_ROOT}/tokenizers"
    TOKENIZER_PATH = f"{TOKENIZER_DIR}/fineweb_1024_bpe.model"

    # --- Step 1: Download dataset if not cached in volume ---
    need_download = (
        not os.path.isdir(DATASET_DIR)
        or len([f for f in os.listdir(DATASET_DIR) if f.endswith(".bin")]) < 2
    )
    if need_download:
        print("=== Downloading dataset from HuggingFace ===", flush=True)
        # Use the repo's own download script
        env = os.environ.copy()
        env["MATCHED_FINEWEB_REPO_ID"] = "willdepueoai/parameter-golf"
        env["MATCHED_FINEWEB_REMOTE_ROOT_PREFIX"] = "datasets"

        # The download script expects to be run from a directory where
        # data/datasets/ and data/tokenizers/ exist relative to it.
        # We'll set it up so it writes into /data/
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(TOKENIZER_DIR, exist_ok=True)

        from huggingface_hub import hf_hub_download

        repo_id = "willdepueoai/parameter-golf"
        remote_prefix = "datasets"

        # Download manifest first
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            filename="manifest.json",
            subfolder=remote_prefix,
            repo_type="dataset",
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Find dataset entry for fineweb10B_sp1024
        dataset_entry = next(
            (x for x in manifest.get("datasets", []) if x.get("name") == "fineweb10B_sp1024"),
            None,
        )
        if dataset_entry is None:
            raise ValueError("fineweb10B_sp1024 not found in manifest")

        stats = dataset_entry.get("stats", {})
        n_train = int(stats.get("files_train", 80))
        n_val = int(stats.get("files_val", 1))
        tokenizer_name = dataset_entry.get("tokenizer_name")

        print(f"Dataset: {n_train} train shards, {n_val} val shards", flush=True)

        # Download all validation shards
        os.makedirs(DATASET_DIR, exist_ok=True)
        for i in range(n_val):
            fname = f"fineweb_val_{i:06d}.bin"
            print(f"  Downloading {fname}...", flush=True)
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                subfolder=f"{remote_prefix}/datasets/fineweb10B_sp1024",
                repo_type="dataset",
            )
            dst = os.path.join(DATASET_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy2(cached, dst)

        # Download all training shards
        for i in range(n_train):
            fname = f"fineweb_train_{i:06d}.bin"
            dst = os.path.join(DATASET_DIR, fname)
            if os.path.exists(dst):
                continue
            print(f"  Downloading {fname}...", flush=True)
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                subfolder=f"{remote_prefix}/datasets/fineweb10B_sp1024",
                repo_type="dataset",
            )
            shutil.copy2(cached, dst)

        # Download tokenizer
        tokenizer_entry = next(
            (x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name),
            None,
        )
        if tokenizer_entry:
            for key in ("model_path", "vocab_path", "path"):
                val = tokenizer_entry.get(key)
                if val:
                    fname = os.path.basename(val)
                    print(f"  Downloading tokenizer: {val}...", flush=True)
                    cached = hf_hub_download(
                        repo_id=repo_id,
                        filename=fname,
                        subfolder=os.path.dirname(f"{remote_prefix}/{val}"),
                        repo_type="dataset",
                    )
                    dst = os.path.join(TOKENIZER_DIR, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(cached, dst)

        vol.commit()
        n_files = len([f for f in os.listdir(DATASET_DIR) if f.endswith(".bin")])
        print(f"=== Dataset ready: {n_files} .bin files ===", flush=True)
    else:
        n_files = len([f for f in os.listdir(DATASET_DIR) if f.endswith(".bin")])
        print(f"=== Dataset cached: {n_files} .bin files ===", flush=True)

    # Verify tokenizer exists
    if not os.path.exists(TOKENIZER_PATH):
        # Try to find it
        for f in os.listdir(TOKENIZER_DIR):
            if f.endswith(".model"):
                TOKENIZER_PATH_ACTUAL = os.path.join(TOKENIZER_DIR, f)
                print(f"Found tokenizer: {TOKENIZER_PATH_ACTUAL}", flush=True)
                break
        else:
            raise FileNotFoundError(f"No tokenizer .model file found in {TOKENIZER_DIR}")
    else:
        TOKENIZER_PATH_ACTUAL = TOKENIZER_PATH

    # --- Step 2: Run training with torchrun ---
    env = os.environ.copy()
    # Use defaults from train_gpt.py; only override paths and essential config
    env.update({
        "RUN_ID": os.environ.get("RUN_ID", "maximalist_batch_v2"),
        "DATA_PATH": DATASET_DIR,
        "TOKENIZER_PATH": TOKENIZER_PATH_ACTUAL,
        "TRAIN_LOG_EVERY": "100",
        "VAL_LOSS_EVERY": "1000",
    })
    # Forward any model/training env vars from caller
    forward_vars = [
        "MODEL_FAMILY", "NUM_LAYERS", "NUM_LOOPS", "MODEL_DIM", "NUM_HEADS",
        "NUM_KV_HEADS", "MLP_MULT", "MLP_HIDDEN", "MPK_K_STRIDE", "MPK_M_STRIDE",
        "TIE_EMBEDDINGS", "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN",
        "MATRIX_LR", "SCALAR_LR", "TIED_EMBED_LR", "EMBED_LR", "HEAD_LR",
        "QUANT_BITS", "MUON_WEIGHT_DECAY", "WARMDOWN_ITERS",
        "EVAL_STRIDE", "EVAL_SEQ_LEN", "EVAL_BATCH_SEQS",
        "GRAD_CLIP_NORM", "SEED",
    ]
    for var in forward_vars:
        val = os.environ.get(var)
        if val is not None:
            env[var] = val
    # Allow debug mode via env var: DEBUG_WALLCLOCK=60 for 1-minute test
    debug_wallclock = os.environ.get("DEBUG_WALLCLOCK", "")
    if debug_wallclock:
        env["MAX_WALLCLOCK_SECONDS"] = debug_wallclock
        env["EVAL_STRIDE"] = "0"  # Skip sliding window eval in debug
        env["VAL_LOSS_EVERY"] = "500"

    print("\n=== Starting training with torchrun (8xH100) ===", flush=True)
    print(f"  DATA_PATH={DATASET_DIR}", flush=True)
    print(f"  TOKENIZER_PATH={TOKENIZER_PATH_ACTUAL}", flush=True)
    t0 = time.time()

    result = subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "/workspace/train_gpt.py",
        ],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd="/workspace",
    )

    elapsed = time.time() - t0
    print(f"\n=== Training completed in {elapsed:.1f}s (return code: {result.returncode}) ===", flush=True)

    # --- Step 3: Save artifacts to volume ---
    artifacts_dir = f"{DATA_ROOT}/results/maximalist_batch"
    os.makedirs(artifacts_dir, exist_ok=True)

    for fname in ["final_model.pt", "final_model.int8.ptz"]:
        src = f"/workspace/{fname}"
        if os.path.exists(src):
            dst = os.path.join(artifacts_dir, fname)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / 1_000_000
            print(f"  Saved {fname}: {size_mb:.2f} MB", flush=True)

    # Save log file
    log_dir = "/workspace/logs"
    if os.path.isdir(log_dir):
        for f in os.listdir(log_dir):
            if f.endswith(".txt"):
                src = os.path.join(log_dir, f)
                dst = os.path.join(artifacts_dir, "train.log")
                shutil.copy2(src, dst)
                print(f"  Saved train.log ({os.path.getsize(src)} bytes)", flush=True)

                # Print last 50 lines of the log
                with open(src) as fh:
                    lines = fh.read().strip().split("\n")
                print("\n=== Last 50 lines of train.log ===")
                for line in lines[-50:]:
                    print(line)

    vol.commit()
    print(f"\n=== Artifacts saved to volume at {artifacts_dir} ===", flush=True)
    return result.returncode


@app.local_entrypoint()
def main():
    print("Launching parameter-golf maximalist batch training on 8xH100...")
    rc = train.remote()
    if rc == 0:
        print("\nTraining succeeded! Results saved to Modal volume 'parameter-golf-data'.")
        print("Download results with: modal volume get parameter-golf-data results/maximalist_batch/ ./results/")
    else:
        print(f"\nTraining failed with return code: {rc}")
        print("Check logs above for errors.")
