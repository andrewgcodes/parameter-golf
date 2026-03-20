This record captures the `Comprehensive V2` submission.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.10`
- SmearGate: enabled (learned adjacent token blending)
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Quantization: int6-in-int8 + zlib-9 compression
- FP16 tied embedding passthrough
- Late-K passthrough: last 1 layer key weights kept in FP16
- RoPE base: 200000
- Warmdown: 3000 iterations
- Evaluation: sliding window with stride=64

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
DATA_PATH=/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=1000 \
QUANT_BITS=6 \
FP16_TIED_EMBED=1 \
LATE_K_LAYERS=1 \
SWA_ENABLED=0 \
USE_BIGRAM_HASH=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `6041/20000` steps due to the wallclock cap.
- Post-quant roundtrip eval: `val_loss:1.9429`, `val_bpb:1.1507`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.15066584`
- Train time: `600015ms` (`step_avg:99.33ms`)
- Serialized model int8+zlib: `15854925 bytes`
- Code size: `56125 bytes`
- Total submission size int8+zlib: `15911050 bytes`

Key techniques:
- Muon optimizer with Newton-Schulz orthogonalization
- 11-layer transformer with GQA (8 heads, 4 KV heads)
- MLP 3x multiplier for increased capacity
- SmearGate for learned adjacent token blending
- Int6-in-int8 quantization with per-row scales
- FP16 tied embedding passthrough (reduces quantization damage to dual-role tensor)
- Late-K passthrough (last 1 layer key weights in FP16)
- Warmdown LR decay (3000 iterations) for quantization-friendly weights
- RoPE with base=200K and NTK-aware dynamic scaling
- Logit softcap at 30.0
- Phase-transition residual mixing
- Orthogonal init with muP output scaling

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
