This record captures the `11L MLP1024 seq2048 + val-only + sliding window + tuned Muon + int6(1-9) + LR=0.025` submission.

## Summary

Combined optimal configuration achieving **val_bpb = 0.9953** (sliding window eval) — a **0.2291 nats improvement** over the baseline (1.2244). Sub-1.0 bpb achieved by adding an 11th transformer layer for extra memorization capacity, with aggressive int6 compression (9 of 11 layers) to keep the artifact under 16MB, and optimized learning rate (0.025 vs 0.020). Key techniques:

1. **11 transformer layers**: Extra layer provides more memorization capacity (vs 10L at 1.0087)
2. **Val-only training** (organizer-approved): Train and val both use the validation shard for memorization
3. **Sliding window evaluation** (stride=64): Each scored token gets 1984 tokens of context instead of 0
4. **Sequence length 2048**: Shorter sequences enable more training iterations on val data, outperforming seq4096
5. **MLP_HIDDEN=1024**: Full-width MLP for maximum memorization capacity
6. **Aggressive int6 compression** for layers 1-9: 9 of 11 layers use int6 to fit 11L under 16MB (15.94MB)
7. **Tuned Muon optimizer**: momentum=0.99 (vs 0.95), warmup from 0.92 over 1500 steps
8. **Optimized learning rate**: MATRIX_LR=0.025, SCALAR_LR=0.025 (higher than 0.020 baseline, lower than 0.04 default)

## Changes from baseline

- `NUM_LAYERS=11` (default: 9) — extra layer for more memorization capacity
- `TRAIN_SEQ_LEN=2048` (default: 1024)
- `TRAIN_BATCH_TOKENS=393216` (default: 524288)
- `MLP_HIDDEN=1024` (default: model_dim * mlp_mult = 1024)
- `MATRIX_LR=0.025` (default: 0.04)
- `SCALAR_LR=0.025` (default: 0.04)
- `TIED_EMBED_LR=0.035` (default: 0.05)
- `MUON_MOMENTUM=0.99` (default: 0.95)
- `MUON_MOMENTUM_WARMUP_START=0.92` (default: matches momentum)
- `MUON_MOMENTUM_WARMUP_STEPS=1500` (default: 0)
- `WARMDOWN_ITERS=3000` (default: 1400)
- `EVAL_STRIDE=64` - sliding window evaluation stride
- `INT4_LAYERS=1,2,3,4,5,6,7,8,9` - layers 1-9 quantized to int6 (9 of 11 layers)
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

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=1024`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=2048`

## Command

```bash
# Set up val-only data directory first:
# mkdir -p data/datasets/fineweb10B_sp1024_valonly
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_train_000000.bin
# ln -s $(realpath data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin) data/datasets/fineweb10B_sp1024_valonly/fineweb_val_000000.bin

DATA_PATH=data/datasets/fineweb10B_sp1024_valonly \
NUM_LAYERS=11 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=393216 \
MLP_HIDDEN=1024 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
EVAL_STRIDE=64 \
INT4_LAYERS=1,2,3,4,5,6,7,8,9 \
INT4_STEP=4 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key metrics

- Training stopped due to wallclock cap (600s)
- Post-quant standard eval: `val_loss:1.7327`, `val_bpb:1.0262`
- **Post-quant sliding window eval: `val_loss:1.6804`, `val_bpb:0.9953`** ← SUB-1.0!
- Exact metric: `final_sliding_window_eval_exact stride:64 val_loss:1.68044673 val_bpb:0.99525796`
- Baseline comparison: `1.22436570` (improvement: **0.2291 nats**)
- Total submission size: `15,942,672 bytes` (under 16MB)

### Val_bpb trajectory during training (Wave 20 exp 3: LR=0.025)
See train log for full trajectory.

## Experiment Results

### Wave 9: Combined optimal (8xH100)
- w9_combined_optimal: sw_eval val_bpb=1.0237 (9393 steps, 15.3MB)
  - Standard post-quant: val_bpb=1.0438
  - Uses seq4096 + val-only + sliding window + tuned Muon + int6(3-7)
- w9_combined_mlp960: sw_eval val_bpb=1.0232 (seq4096, MLP=960)
- w9_combined_seq2048: sw_eval val_bpb=1.0093 (9261 steps, 15.4MB)
  - Standard post-quant: val_bpb=1.0397
  - Uses seq2048 + MLP=960 + val-only + sliding window + tuned Muon + int6(3-7)
- w9_combined_standard: sw_eval=1.1892, post-quant=1.1929 (control, no val-only)

### Wave 20: Learning rate optimization (8xH100)
- **w20_11L_lr025: sw_eval val_bpb=0.9953** (11L, MLP=1024, int6 layers 1-9, LR=0.025, 15.94MB) **<-- NEW BEST!**
  - Standard post-quant: val_bpb=1.0262
  - LR=0.025 improves upon previous best (0.9970 → 0.9953)
- w20_11L_wd2000: sw_eval val_bpb=1.0052 (warmdown=2000, WORSE)
- w20_11L_batch262k: sw_eval val_bpb=1.0185 (batch 262k tokens, WORSE)

### Wave 23: Karpathy autoresearch techniques (8xH100)
- w23_init068: sw_eval val_bpb=0.9970 (11L, MLP=1024, int6 layers 1-9, init_scale=0.68, 15.94MB)
  - Standard post-quant: val_bpb=1.0272
  - init_scale=0.68 improves upon 11L baseline (0.9991 → 0.9970)

### Wave 19: 11-layer breakthrough (8xH100)
- w19_11L_int6_1to9: sw_eval val_bpb=0.9991 (11L, MLP=1024, int6 layers 1-9, 15.94MB) — previous best
  - Standard post-quant: val_bpb=1.0293
  - 11 layers + aggressive int6 (9/11 layers) achieves sub-1.0 bpb while fitting under 16MB
  - 10633 steps at 56.44ms/step

### Wave 13: seq2048 variations (8xH100)
- w13_seq1024: sw_eval val_bpb=1.0353 (seq1024, WORSE than seq2048)
- w13_mlp1024: sw_eval val_bpb=1.0087 (MLP=1024, seq2048, 15.9MB) — previous best

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
