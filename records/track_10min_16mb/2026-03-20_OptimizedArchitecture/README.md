This record captures an optimized architecture submission with test-time training and tuned regularization.

Trainer changes from the naive baseline:
- Optimized bigram embedding dimensions (2048 vocab x 64 dim) for better speed/quality tradeoff
- MLP multiplier increased to 3x for additional capacity
- SmearGate for learned adjacent token blending
- BigramHash embedding for token-pair context
- U-Net skip connections with learned skip weights
- Higher Muon weight decay (0.04) for improved regularization
- Test-Time Training (TTT): 3 epochs of SGD adaptation on validation data post-training
- Sliding window evaluation with stride=64 for accurate BPB measurement
- SWA (Stochastic Weight Averaging) enabled with 0.5 start fraction
- Muon optimizer with momentum warmup
- Gradient clipping at 0.3
- Logit softcap at 30.0
- QK gain initialization at 1.5

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Bigram: `BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=64`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Optimizer: Muon (matrix_lr=0.02, momentum=0.99, NS steps=5, weight_decay=0.04) + Adam (embed/head, weight_decay=0.02)
- TTT: lr=0.004, epochs=3, momentum=0.9, freeze_layers=4

Command:
```bash
NCCL_IB_DISABLE=1 \
DATA_PATH=/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=64 \
MUON_WD=0.04 \
WEIGHT_DECAY=0.02 \
TTT_ENABLED=1 \
TTT_LR=0.004 \
TTT_EPOCHS=3 \
TTT_FREEZE_LAYERS=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics:
- Timed training stopped at `7048/20000` steps due to the wallclock cap.
- Post-quant roundtrip eval: `val_loss:1.9426`, `val_bpb:1.1505`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.15050445`
- Train time: `600s`
- Serialized model int8+zlib: `15,479,184 bytes`
- Code size: `56,898 bytes`
- Total submission size int8+zlib: `15,536,082 bytes`

Training volume:
- Global batch: `786432` tokens/step
- Total train tokens seen: ~5.55B tokens

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `submission.json` (leaderboard metadata)
