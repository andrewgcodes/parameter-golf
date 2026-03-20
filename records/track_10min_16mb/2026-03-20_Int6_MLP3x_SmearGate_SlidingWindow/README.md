# Int6 MLP3x + SmearGate + SlidingWindow + SWA

**val_bpb: 1.1439** (seed 42, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
# All parameters are set as defaults in train_gpt.py. No env overrides needed:
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_SlidingWindow/train_gpt.py

# With specific seed:
SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_SlidingWindow/train_gpt.py
```

## Seed Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact (bytes) |
|------|----------|---------|-------|---------|-------------------|
| 42   | 1.93137471 | 1.14387192 | 6412 | 93.58 | 15,423,789 |
| 1337 | 1.93311411 | 1.14490209 | 5929 | 101.18 | 15,423,789 |
| 7    | 1.93330927 | 1.14501768 | 5952 | 100.76 | 15,423,789 |

**Mean val_bpb: 1.14459723** (std: 0.00063079, 95% CI: [1.14388, 1.14531])

## Key Techniques

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights (most compressible, high zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive)
- **FP16** for tied embeddings and last-layer key projections (c_k)
- Per-row quantization scales (fp16) for 2D tensors, single scale for 1D
- zstd-22 compression on quantized weights

### 10-Layer MLP 3x Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536) with relu^2 activation
- U-Net skip connections between layers
- Tied input/output embeddings

### SmearGate
- Learned adjacent token blending gate
- Mixes current token embedding with previous token via sigmoid gate
- Provides cheap local context without additional attention

### Training Optimizations
- Muon optimizer for matrix params (Newton-Schulz orthogonalization)
- AdamW for embeddings and scalar params
- Decoupled weight decay (WD=0.04) applied in Muon step()
- Momentum warmup: 0.92 -> 0.99 over 1500 steps
- Gradient clipping at 0.3
- Warmdown schedule over last 3000 iterations
- seq_len=2048, batch=786K tokens

### Stochastic Weight Averaging (SWA)
- Checkpoints collected during last 40% of warmdown (start_frac=0.4)
- Every 50 steps, 24 checkpoints averaged
- Improves quantization robustness

### 3% Magnitude Pruning
- Zero out smallest 3% of weights before quantization
- Improves zstd compression ratio with negligible quality loss

### Sliding Window Evaluation
- Stride=64, scoring only the strided portion of each window
- Every token scored with near-maximum context (up to 2048)
- ~0.02 BPB improvement vs standard left-to-right evaluation
- Causal/online: no information leakage, each token scored using only prior context

### Orthogonal Initialization
- SVD-based spectral init with power-law scaling (k^-0.5)
- muP-scaled output projections

## Architecture Details

| Component | Value |
|-----------|-------|
| Layers | 10 |
| Model dim | 512 |
| MLP hidden | 1536 (3x) |
| Heads | 8 |
| KV heads | 4 (GQA) |
| Vocab size | 1024 |
| Seq length | 2048 |
| Total params | 25,517,137 |

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer (matrix) | Muon (lr=0.02, momentum=0.99, WD=0.04) |
| Optimizer (embed/scalar) | AdamW (lr=0.02/0.03, WD=0.04) |
| Batch size | 786,432 tokens |
| Warmup steps | 20 |
| Warmdown iters | 3000 |
| Grad clip | 0.3 |
| Training time | 600s wallclock on 8xH100 |
| Steps completed | ~6412 |
| ms/step | ~93.5 |
| SWA | start_frac=0.4, every=50 steps |

## Artifact Size

| Component | Bytes |
|-----------|-------|
| Model (int5/int6+zstd) | 15,370,858 |
| Code | 52,931 |
| **Total** | **15,423,789** |

Fits well under the 16,000,000 byte limit with 576,211 bytes to spare.

## Evaluation

- Sliding window eval with stride=64, batch_seqs=32
- Eval time: ~169s on 8xH100 (within 10-minute limit)
- No training on validation set (pure inference with torch.inference_mode())
- No external downloads during evaluation

## Rule Compliance

1. **No val-set training**: Validation tokens only used in eval functions with `torch.inference_mode()`. No gradient updates on val data.
2. **Causal/online evaluation**: Sliding window scores each token using only prior context. No information leakage.
3. **Training time**: 600s wallclock on 8xH100 (within 10-minute limit).
4. **Eval time**: ~169s (within 10-minute limit).
5. **Artifact size**: 15,423,789 bytes (under 16,000,000 byte limit).
6. **Self-contained**: No external downloads or network calls during evaluation.
7. **Reproducible**: Fixed seed (42), deterministic training.
