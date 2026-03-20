Maximalist Batch + Curriculum Learning submission.

Trainer changes from baseline:
- Doubled batch size: `TRAIN_BATCH_TOKENS=1048576` (1M tokens/step vs 524K default)
- Curriculum learning: progressive sequence length (1024 -> 2048 -> target) based on wallclock time
- 10 transformer layers (vs 9 baseline)
- MLP 3x expansion (`MLP_MULT=3`)
- Int6 quantization for transformer blocks (int8 for embeddings) to fit more params in 16MB
- FP16 tied embedding passthrough in quantization (reduces quant damage on shared weight)
- Sliding window evaluation with stride=64
- Overtone spectral embedding init (SVD power-law spectrum shaping)
- Phase-transition residual mixing initialization
- Muon optimizer with weight decay (0.02)
- Aggressive warmdown (`WARMDOWN_ITERS=20000` = always decaying)
- Gradient clipping (norm=1.0)
- U-Net skip connections from encoder to decoder layers

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=1048576 TRAIN_SEQ_LEN=2048`
- Curriculum: `CURRICULUM_ENABLED=1 CURRICULUM_START_SEQ=1024 CURRICULUM_MID_SEQ=2048`
- Quantization: `QUANT_BITS=6` for blocks, int8 for other tensors, FP16 for embeddings
- Eval: `EVAL_SEQ_LEN=2048 EVAL_STRIDE=64`

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=maximalist_batch_curriculum \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
NUM_LAYERS=10 \
MLP_MULT=3 \
TRAIN_BATCH_TOKENS=1048576 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
WARMDOWN_ITERS=20000 \
CURRICULUM_ENABLED=1 \
QUANT_BITS=6 \
MUON_WEIGHT_DECAY=0.02 \
GRAD_CLIP_NORM=1.0 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key techniques stacked:
1. **Larger batch** (2x): More tokens per gradient update = better gradient estimates, more data coverage
2. **Curriculum learning**: Short seqs first (high throughput), longer seqs later (better context)
3. **Int6 quantization**: Smaller quantized weights = more parameters fit in 16MB budget
4. **MLP 3x**: Wider feed-forward layers enabled by int6 compression savings
5. **10 layers**: Extra depth for more representational capacity
6. **Sliding window eval**: Each token scored with near-full context (stride=64)
7. **Overtone init**: Embedding spectrum shaped to power-law decay for better initial representations
8. **Phase-transition resid_mix**: Early layers trust embedding more, late layers trust residual more
9. **Aggressive warmdown**: Always-decaying LR produces tighter weight distributions with fewer outliers
10. **Muon weight decay**: Regularization via decoupled weight decay in the Muon optimizer

Included files:
- `train_gpt.py` (code snapshot for the run)
