This record captures an optimized architecture submission.

Trainer changes from the naive baseline:
- Optimized bigram embedding dimensions (2048 vocab × 64 dim) for better speed/quality tradeoff
- MLP multiplier increased to 3x for additional capacity
- SmearGate for learned adjacent token blending
- Sliding window evaluation with stride=64 for accurate BPB measurement
- SWA (Stochastic Weight Averaging) enabled with 0.5 start fraction
- Muon optimizer with momentum warmup (0.92→0.99 over 1500 steps)
- Gradient clipping at 0.3

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Bigram: `BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=64`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Optimizer: Muon (matrix_lr=0.02, momentum=0.99, NS steps=5) + Adam (embed/head)

Command:
```bash
NCCL_IB_DISABLE=1 \
DATA_PATH=/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics:
- Timed training stopped at `7097/20000` steps due to the wallclock cap.
- Post-quant roundtrip eval: `val_loss:1.9494`, `val_bpb:1.1546`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.15457352`
- Train time: `600s` (`step_avg:84.52ms`)
- Serialized model int8+zlib: `15,942,927 bytes`
- Code size: `52,954 bytes`
- Total submission size int8+zlib: `15,995,881 bytes`

Training volume:
- Global batch: `786432` tokens/step
- Total train tokens seen: ~5.58B tokens

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `submission.json` (leaderboard metadata)
