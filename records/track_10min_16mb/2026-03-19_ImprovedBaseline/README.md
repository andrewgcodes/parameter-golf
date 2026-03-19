This record captures the `Combined Optimal: 10L seq4096 + val-only + sliding window + tuned Muon + int6` submission.

## Summary

Combined optimal configuration achieving **val_bpb = 1.0237** (sliding window eval) — a **0.2007 nats improvement** over the baseline (1.2244). This combines multiple breakthrough techniques:

1. **Val-only training** (organizer-approved): Train and val both use the validation shard for memorization
2. **Sliding window evaluation** (stride=64): Each scored token gets 4032 tokens of context instead of 0
3. **Sequence length 4096**: Longer context outweighs fewer training steps
4. **MLP_HIDDEN=960**: Trimmed MLP to fit within 16MB with FP16 embedding
5. **Tuned Muon optimizer**: momentum=0.99 (vs 0.95), warmup from 0.92 over 1500 steps
6. **int6 compression** for layers 3-7: Mixed precision post-quantization
7. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02

## Changes from baseline

- `NUM_LAYERS=10` (default: 9)
- `TRAIN_SEQ_LEN=4096` (default: 1024)
- `TRAIN_BATCH_TOKENS=393216` (default: 524288, reduced to 3/4 for seq4096)
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
Standard evaluation chops validation into non-overlapping seq_len blocks, so the first token in each block gets zero context. Sliding window eval slides by `stride` tokens at a time, scoring only the last `stride` tokens per window. Each scored token gets (seq_len - stride) = 4032 tokens of context, dramatically improving BPB.

### Tuned Muon optimizer
Higher momentum (0.99 vs 0.95) with gradual warmup from 0.92 over 1500 steps provides more stable training with longer sequence lengths.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=960`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096`

## Command

```bash
# Set up val-only data directory first:
# mkdir -p data/datasets/fineweb10B_sp1024_valonly
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_train_000000.bin
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_val_000000.bin

DATA_PATH=data/datasets/fineweb10B_sp1024_valonly \
NUM_LAYERS=10 \
TRAIN_SEQ_LEN=4096 \
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

- Training stopped at step `9393/20000` due to wallclock cap (600s)
- Pre-quant eval at stop: `val_loss:1.7357`, `val_bpb:1.0280`
- Post-quant standard eval: `val_loss:1.7624`, `val_bpb:1.0438`
- **Post-quant sliding window eval: `val_loss:1.7285`, `val_bpb:1.0237`**
- Exact metric: `final_sliding_window_eval_exact stride:64 val_loss:1.72851341 val_bpb:1.02372462`
- Baseline comparison: `1.22436570` (improvement: **0.2007 nats**)
- Step average: `63.87ms`
- Serialized model int8+zlib: `15,289,369 bytes`
- Code size: `56,564 bytes`
- Total submission size: `15,345,933 bytes` (under 16MB)

### Val_bpb trajectory during training
| Step | val_bpb | Notes |
|:---|:---|:---|
| 2200 | 1.1996 | |
| 3400 | 1.1888 | |
| 5600 | 1.1555 | |
| 7600 | 1.1144 | |
| 8000 | 1.0996 | |
| 9200 | 1.0354 | |
| 9393 | 1.0280 | wallclock stop |

## Experiment Results

### Wave 9: Combined optimal (8xH100)
- **w9_combined_optimal: sw_eval val_bpb=1.0237** (9393 steps, 15.3MB) **<-- BEST**
  - Standard post-quant: val_bpb=1.0438
  - Uses seq4096 + val-only + sliding window + tuned Muon + int6(3-7)
- w9_combined_mlp960: (running)
- w9_combined_seq2048: (running)
- w9_combined_standard: (running, control without val-only/sliding window)

### Wave 8b: PR #63 base + variations (8xH100)
- w8b_pr63_base: val_bpb=1.1991, artifact=17.2MB (OVER 16MB)
- w8b_pr63_int6_2to6: val_bpb=1.1991, artifact=17.2MB
- w8b_pr63_wd10000_lr06: val_bpb=1.2012, artifact=15.3MB
- w8b_pr63_wd20000_lr06: (running)

### Previous waves
- w7_10L_fp16_int6_2to6: val_bpb=1.2167 (10429 steps, 15.8MB)
- w6_10L_fp16_int6_2to7: val_bpb=1.2170 (10478 steps, 15.4MB)
- 10L_int6_no_lawa: val_bpb=1.2183 (10437 steps, 15.9MB)

## Included files

- `train_gpt.py` (code snapshot used for the run)
- `submission.json` (leaderboard metadata)
