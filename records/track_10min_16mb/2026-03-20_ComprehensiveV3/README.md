# Comprehensive V3: 10L Int6 Mixed Quant + SmearGate + BigramHash + SWA + Muon WD

**val_bpb: 1.1439** (seed 42, sliding window eval stride=64)

## Architecture
- 10-layer transformer, dim=512, 8 heads (4 KV heads via GQA)
- MLP 3x expansion (1536 hidden dim)
- SmearGate bigram embedding + BigramHash (vocab=10240, dim=128)
- RoPE positional encoding
- Tied input/output embeddings

## Training
- Muon optimizer with WD=0.04, momentum=0.99
- Batch size: 786,432 tokens
- Warmdown: 3000 iters
- SWA: start_frac=0.4, every=50 steps
- 600s wallclock on 8xH100

## Quantization & Compression
- Int5 for MLP weights (clip_range=15)
- Int6 for attention weights (clip_range=31)
- FP16 passthrough for tied embeddings and last-layer c_k
- zstd-22 compression
- Total artifact: 15,423,789 bytes (under 16MB)

## Evaluation
- Sliding window eval with stride=64
- Tokenizer-agnostic BPB metric
