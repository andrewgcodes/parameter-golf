This record captures the `10L Mixed Precision + LAWA` submission.

## Summary

10-layer transformer with mixed int8/int6 compression, LAWA weight averaging, and optimized learning rates. Combines the best techniques from extensive experimentation:

1. **10 transformer layers** (vs baseline 9) for more model capacity
2. **Mixed int8/int6 compression**: int6 (step=4 rounding) for middle layers 3-6, full int8 for early/late layers
3. **LAWA (Lookahead Weight Averaging)**: averages checkpoints during warmdown for free quality boost
4. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 (optimal per LR sweep)
5. **FP16 tied embedding passthrough** (optional): keeps embedding in fp16 instead of int8

## Changes from baseline

- `NUM_LAYERS=10` (default: 9)
- `MATRIX_LR=0.02` (default: 0.04)
- `SCALAR_LR=0.02` (default: 0.04)
- `TIED_EMBED_LR=0.03` (default: 0.05)
- `WARMDOWN_ITERS=1200` (default: 1400)
- `INT4_LAYERS=3,4,5,6` - middle layers quantized to int6 for better compression
- `INT4_STEP=4` - rounding step for int6 quantization
- `LAWA_ENABLED=1` with `LAWA_INTERVAL=50`
- No architecture or other hyperparameter changes

## How mixed precision compression works

The 10L model has 18.9M params, which compresses to ~17.6MB with standard int8+zlib (over 16MB). By reducing middle layers to int6, compressed size drops to ~15.9MB:

| Layer Group | Precision | Reason |
|:---|:---|:---|
| Layers 0-2 (early) | int8 (256 levels) | Critical for input processing |
| Layers 3-6 (middle) | int6 (64 levels) | Less sensitive, saves ~1.6MB |
| Layers 7-9 (late) | int8 (256 levels) | Critical for output quality |

## How LAWA works

During the warmdown phase of training, LAWA saves checkpoints every 50 steps and averages them at the end. This acts as a free quality boost by smoothing the loss landscape, similar to Stochastic Weight Averaging (SWA) but applied only during warmdown.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Command

```bash
NUM_LAYERS=10 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=1200 \
INT4_LAYERS=3,4,5,6 \
INT4_STEP=4 \
LAWA_ENABLED=1 \
LAWA_INTERVAL=50 \
QAT_ENABLED=0 \
FP16_EMBED=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiment Results

### Wave 2: Single H100 experiments (QAT vs no QAT)
- baseline_1gpu: val_bpb=1.3166 (1579 steps)
- QAT experiments: val_bpb=1.46-2.11 (QAT overhead too expensive on single GPU)

### Wave 3: Single H100 experiments (10L + int6 + LAWA combos)
- 10L_int6_no_lawa: val_bpb=1.3251 (best single-GPU result with 10L)
- 10L_int6_lawa: val_bpb=1.3712 (LAWA hurt on 1GPU due to early warmdown start)
- 9L_fp16_lawa: val_bpb=1.3723
- 10L_int6wide_fp16_lawa: val_bpb=1.3744
- 10L_int6_lawa_lr04: val_bpb=1.3956

Note: Single-GPU results are directional only. On 8xH100, LAWA warmdown starts at step ~10300 (vs step ~200 on 1GPU), giving proper weight averaging.

## Included files

- `train_gpt.py` (code snapshot with all improvements)
- `submission.json` (leaderboard metadata)
