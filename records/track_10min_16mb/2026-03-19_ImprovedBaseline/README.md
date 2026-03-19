This record captures the `10L Mixed Precision (int6)` submission.

## Summary

10-layer transformer with mixed int8/int6 compression and optimized learning rates. LAWA was tested but found to increase the quantization gap, so it is disabled. Combines the best techniques from extensive experimentation:

1. **10 transformer layers** (vs baseline 9) for more model capacity
2. **Mixed int8/int6 compression**: int6 (step=4 rounding) for middle layers 3-6, full int8 for early/late layers
3. **Lower learning rates**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03 (optimal per LR sweep)

## Changes from baseline

- `NUM_LAYERS=10` (default: 9)
- `MATRIX_LR=0.02` (default: 0.04)
- `SCALAR_LR=0.02` (default: 0.04)
- `TIED_EMBED_LR=0.03` (default: 0.05)
- `WARMDOWN_ITERS=1200` (default: 1400)
- `INT4_LAYERS=3,4,5,6` - middle layers quantized to int6 for better compression
- `INT4_STEP=4` - rounding step for int6 quantization
- `LAWA_ENABLED=0` (LAWA increases quantization gap by ~0.001 bpb)
- No architecture or other hyperparameter changes

## How mixed precision compression works

The 10L model has 18.9M params, which compresses to ~17.6MB with standard int8+zlib (over 16MB). By reducing middle layers to int6, compressed size drops to ~15.9MB:

| Layer Group | Precision | Reason |
|:---|:---|:---|
| Layers 0-2 (early) | int8 (256 levels) | Critical for input processing |
| Layers 3-6 (middle) | int6 (64 levels) | Less sensitive, saves ~1.6MB |
| Layers 7-9 (late) | int8 (256 levels) | Critical for output quality |

## LAWA Finding

LAWA (Lookahead Weight Averaging) was tested but found to **hurt** post-quantization performance:
- With LAWA: val_bpb = 1.2196 (quant gap: 0.0061)
- Without LAWA: val_bpb = 1.2183 (quant gap: 0.0052)

LAWA averaging smooths weights in a way that increases the quantization gap. Disabled for final submission.

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
LAWA_ENABLED=0 \
QAT_ENABLED=0 \
FP16_EMBED=0 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key metrics (from `train.log`)

- Timed training stopped at `10437/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0483`, `val_bpb:1.2131`
- Post-quant roundtrip eval: `val_loss:2.0571`, `val_bpb:1.2183`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.21831774`
- Baseline comparison: `1.22436570` (improvement: **0.00605 nats**)
- Train time: `599935ms` (`step_avg:57.48ms`)
- Peak memory: `13631 MiB allocated`, `14654 MiB reserved`
- Serialized model int8+zlib: `15866382 bytes`
- Code size: `54721 bytes`
- Total submission size int8+zlib: `15921103 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `5473034240`

## Experiment Results

### 8xH100 validation (final)
- **10L_int6_no_lawa: val_bpb=1.21831774** (10437 steps, 15.9MB artifact) **<-- best**
- 10L_int6_lawa: val_bpb=1.21963035 (10386 steps, 15.9MB artifact)

### Wave 2: Single H100 experiments (QAT vs no QAT)
- baseline_1gpu: val_bpb=1.3166 (1579 steps)
- QAT experiments: val_bpb=1.46-2.11 (QAT overhead too expensive on single GPU)

### Wave 3: Single H100 experiments (10L + int6 + LAWA combos)
- 10L_int6_no_lawa: val_bpb=1.3251 (best single-GPU result with 10L)
- 10L_int6_lawa: val_bpb=1.3712 (LAWA hurt on 1GPU due to early warmdown start)
- 9L_fp16_lawa: val_bpb=1.3723
- 10L_int6wide_fp16_lawa: val_bpb=1.3744
- 10L_int6_lawa_lr04: val_bpb=1.3956

Note: Single-GPU results are directional only. On 8xH100, training runs ~10400 steps vs ~1400 on 1GPU.

## Included files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
