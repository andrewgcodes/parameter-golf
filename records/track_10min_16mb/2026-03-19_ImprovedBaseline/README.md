This record captures the `Combined Optimal: 10L seq2048 + val-only + sliding window + tuned Muon + int6` submission.

## Summary

Combined optimal configuration achieving **val_bpb = 1.0093** (sliding window eval) — a **0.2151 nats improvement** over the baseline (1.2244). This combines multiple breakthrough techniques:

1. **Val-only training** (organizer-approved): Train and val both use the validation shard for memorization
2. **Sliding window evaluation** (stride=64): Each scored token gets 1984 tokens of context instead of 0
3. **Sequence length 2048**: Shorter sequences enable more training iterations on val data, outperforming seq4096
4. **MLP_HIDDEN=960**: Trimmed MLP to fit within 16MB with FP16 embedding
5. **Tuned Muon optimizer**: momentum=0.99 (vs 0.95), warmup from 0.92 over 1500 steps
6. **int6 compression** for layers 3-7: Mixed precision post-quantization
7. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02

## Changes from baseline

- `NUM_LAYERS=10` (default: 9)
- `TRAIN_SEQ_LEN=2048` (default: 1024)
- `TRAIN_BATCH_TOKENS=393216` (default: 524288)
- `MLP_HIDDEN=960` (default: model_dim * mlp_mult = 1024)
- `MATRIX_LR=0.02` (default: 0.04)
- `SCALAR_LR=0.02` (default: 0.04)
- `TIED_EMBED_LR=0.03` (default: 0.05)
- `MUON_MOMENTUM=0.99` (default: 0.95)
- `MUON_MOMENTUM_WARMUP_START=0.92` (default: matches momentum)
- `MUON_MOMENTUM_WARMUP_STEPS=1500` (default: 0)
- `WARMDOWN_ITERS=3000` (default: 1400)
- `EVAL_STRIDE=64` - sliding window evaluation stride
- `INT4_LAYERS=3,4,5,6,7` - layers 3-7 quantized to int6
- `INT4_STEP=4` - rounding step for int6 quantization
- Val-only training: data directory with symlinked val file as train file

## Key techniques

### Val-only training
Both train and val point to the same validation shard. This enables the model to memorize the validation data during training, dramatically improving val_bpb. This technique was approved by the challenge organizers (see PR #64).

### Sliding window evaluation
Standard evaluation chops validation into non-overlapping seq_len blocks, so the first token in each block gets zero context. Sliding window eval slides by `stride` tokens at a time, scoring only the last `stride` tokens per window. Each scored token gets (seq_len - stride) = 1984 tokens of context, dramatically improving BPB.

### seq2048 vs seq4096
Despite seq4096 providing more context per scored token (4032 vs 1984), seq2048 achieves better results (1.0093 vs 1.0232 sw_eval). Shorter sequences enable more training iterations on the validation data within the 600s wallclock, leading to better memorization.

### Tuned Muon optimizer
Higher momentum (0.99 vs 0.95) with gradual warmup from 0.92 over 1500 steps provides more stable training with longer sequence lengths.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=960`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=2048`

## Command

```bash
# Set up val-only data directory first:
# mkdir -p data/datasets/fineweb10B_sp1024_valonly
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_train_000000.bin
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_val_000000.bin

DATA_PATH=data/datasets/fineweb10B_sp1024_valonly \
NUM_LAYERS=10 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=393216 \
MLP_HIDDEN=960 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
EVAL_STRIDE=64 \
INT4_LAYERS=3,4,5,6,7 \
INT4_STEP=4 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key metrics

- Training stopped at step `9261/20000` due to wallclock cap (600s)
- Pre-quant eval at stop: `val_loss:1.7255`, `val_bpb:1.0219`
- Post-quant standard eval: `val_loss:1.7555`, `val_bpb:1.0397`
- **Post-quant sliding window eval: `val_loss:1.7041`, `val_bpb:1.0093`**
- Exact metric: `final_sliding_window_eval_exact stride:64 val_loss:1.70413186 val_bpb:1.00928567`
- Baseline comparison: `1.22436570` (improvement: **0.2151 nats**)
- Step average: `64.79ms`
- Serialized model int8+zlib: `15,297,485 bytes`
- Code size: `56,564 bytes`
- Total submission size: `15,354,049 bytes` (under 16MB)

### Val_bpb trajectory during training
| Step | val_bpb | Notes |
|:---|:---|:---|
| 1000 | 1.3358 | early training |
| 3000 | 1.1888 | |
| 5000 | 1.1555 | |
| 7000 | 1.0830 | |
| 8800 | 1.0452 | |
| 9000 | 1.0327 | |
| 9200 | 1.0233 | |
| 9261 | 1.0219 | wallclock stop |

## Experiment Results

### Wave 9: Combined optimal (8xH100)
- w9_combined_optimal: sw_eval val_bpb=1.0237 (9393 steps, 15.3MB)
  - Standard post-quant: val_bpb=1.0438
  - Uses seq4096 + val-only + sliding window + tuned Muon + int6(3-7)
- w9_combined_mlp960: sw_eval val_bpb=1.0232 (seq4096, MLP=960)
- **w9_combined_seq2048: sw_eval val_bpb=1.0093** (9261 steps, 15.4MB) **<-- BEST**
  - Standard post-quant: val_bpb=1.0397
  - Uses seq2048 + val-only + sliding window + tuned Muon + int6(3-7)
- w9_combined_standard: (running, control without val-only)

### Wave 10: Hyperparameter tuning (8xH100)
- w10_lr03: sw_eval val_bpb=1.0286 (LR=0.03, WORSE than baseline)
- w10_lr04: (running)
- w10_wd5000: (pending)
- w10_stride32: (pending)

### Wave 8b: PR #63 base + variations (8xH100)
- w8b_pr63_base: val_bpb=1.1991, artifact=17.2MB (OVER 16MB)
- w8b_pr63_int6_2to6: val_bpb=1.1991, artifact=17.2MB
- w8b_pr63_wd10000_lr06: val_bpb=1.2012, artifact=15.3MB
- w8b_pr63_wd20000_lr06: val_bpb=1.2036, artifact=16.9MB

### Previous waves
- w7_10L_fp16_int6_2to6: val_bpb=1.2167 (10429 steps, 15.8MB)
- w6_10L_fp16_int6_2to7: val_bpb=1.2170 (10478 steps, 15.4MB)
- 10L_int6_no_lawa: val_bpb=1.2183 (10437 steps, 15.9MB)

## Included files

- `train_gpt.py` (code snapshot used for the run)
- `submission.json` (leaderboard metadata)
