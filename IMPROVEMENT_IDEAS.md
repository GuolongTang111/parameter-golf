# Parameter Golf: Improvement Ideas Research

> Current SOTA: **val_bpb = 1.1428** (10L Int5-MLP + BigramHash + SWA + MuonWD)
> Constraint: 16MB artifact, 10-minute training on 8xH100

---

## Table of Contents
1. [What's Working (Current Top Techniques)](#1-whats-working-current-top-techniques)
2. [Untried / Underexplored Ideas](#2-untried--underexplored-ideas)
3. [Architecture Innovations](#3-architecture-innovations)
4. [Quantization & Compression Frontiers](#4-quantization--compression-frontiers)
5. [Training & Optimization](#5-training--optimization)
6. [Evaluation-Time Improvements](#6-evaluation-time-improvements)
7. [Negative Results (What Failed)](#7-negative-results-what-failed)
8. [Prioritized Roadmap](#8-prioritized-roadmap)

---

## 1. What's Working (Current Top Techniques)

These are proven techniques from the leaderboard top 5:

| Technique | Impact (bpb) | Used By |
|-----------|-------------|---------|
| Mixed Int5 MLP / Int6 Attn quantization | -0.003 | SOTA #1 |
| BigramHash embedding (10240 buckets) | -0.002 | SOTA #1, #2 |
| Stochastic Weight Averaging (every 50 steps) | -0.0006 | SOTA #1, #2 |
| SmearGate (previous-token blending) | ~-0.001 | #2, #4 |
| 3x MLP expansion (1536 hidden) | major | #1-#5 |
| Muon optimizer + WD=0.04 | major | #1, #2, #4 |
| Sliding window eval (stride=64) | -0.034 | #3, #4 |
| Int6 QAT via STE | eliminates quant gap | #3, #4 |
| Orthogonal initialization | faster convergence | #1, #2, #4 |
| zstd-22 compression | ~5% better than zlib | #1, #2, #3 |
| FP16 tied embeddings | -0.005 quant penalty | 6+ subs |
| 10-11 transformer layers | -0.003 | #1, #3 |

---

## 2. Untried / Underexplored Ideas

### 2.1 Int4 Quantization for MLP Weights
**Rationale**: SOTA already uses Int5 for MLP (1.88x zstd ratio). Going to Int4 ([-8, 7], 16 levels) could achieve ~2.2-2.5x compression, freeing ~1-2MB for an 11th or 12th layer.

**Risk**: Quality degradation at 16 levels may outweigh parameter count gains. Would need QAT to compensate.

**Experiment**: Try Int4 QAT on MLP weights only, keep Int6 for attention. Measure if the freed bytes can fund enough extra depth to compensate.

### 2.2 Per-Channel / Group Quantization
**Rationale**: Current approach uses per-row scales. Per-group quantization (e.g., groups of 32 or 64 within each row) adds more scale parameters but captures finer weight distribution details.

**Expected gain**: 0.001-0.003 bpb from reduced quantization error, at cost of ~50-100KB extra scale storage.

**Reference**: GPTQ and AWQ use group-wise quantization effectively. QuIP# uses random orthogonal rotations before quantization to make weight distributions more uniform.

### 2.3 Weight Matrix Rotation Before Quantization (QuIP#-style)
**Rationale**: Apply random Hadamard rotation to weight matrices before quantization. This spreads outlier values across all entries, making the distribution more uniform and easier to quantize at low bit-widths.

**Implementation**: `W_rotated = H @ W @ H.T` where H is a Hadamard matrix. Reverse at inference with `W = H.T @ W_rotated @ H`. The Hadamard multiply is O(n log n) via fast Walsh-Hadamard transform.

**Expected gain**: Could enable Int4 for all weights (not just MLP) while maintaining Int6-level quality.

### 2.4 Learned Codebook Quantization (AQLM-style)
**Rationale**: Instead of uniform scalar quantization, learn a codebook of weight vectors. Each group of weights maps to a codebook entry. With 256 entries (1 byte per group), this can represent complex distributions better than uniform bins.

**Risk**: Training the codebook within 10 minutes may not converge. Could work as a post-training step.

### 2.5 Knowledge Distillation from Larger Run
**Rationale**: The non-record track shows a 4-hour run achieves bpb=1.2074. If we train a larger model for 4 hours, we could use it as a teacher for the 10-minute student run.

**Challenge**: Rules say "no external compute during training" but a pre-computed teacher's logits could potentially be included in the 16MB artifact as compressed soft targets. Need to check rules carefully.

### 2.6 Mixture of Experts (MoE)
**Rationale**: MoE allows more total parameters with the same FLOPs per token. With 2 experts and top-1 routing, you double MLP capacity while keeping compute constant.

**Constraint fit**: Router adds minimal parameters. Expert weights quantize independently. The key question is whether 2x MLP capacity at int5 fits in 16MB.

**Expected gain**: Could be significant if compute-bound (not parameter-bound). 10-minute wall clock may be the binding constraint.

### 2.7 Sparse Attention Patterns
**Rationale**: Local + global attention (e.g., every 4th layer attends globally, others use sliding window) could allow longer effective context with less compute per step, enabling more training steps.

### 2.8 Dynamic Sequence Length Curriculum
**Rationale**: Start training with shorter sequences (256 or 512) for fast initial learning, then increase to 2048 for later stages. More tokens processed early, better long-range modeling later.

**Expected gain**: 10-20% more total tokens seen in 10 minutes, with minimal quality loss from short early sequences.

### 2.9 Byte-Level or Sub-Word Tokenizer Optimization
**Rationale**: Current tokenizer is SentencePiece with 1024 vocab. The BPB metric is tokenizer-agnostic, but tokenizer choice affects:
- Embedding table size (vocab_size x dim)
- Sequence length needed to cover same text
- Compression of embedding weights

A smaller vocab (512 or 256) reduces embedding parameters but needs longer sequences. A larger vocab (2048, 4096) shortens sequences but costs more embedding parameters. Optimal may differ from current 1024.

### 2.10 Embedding Factorization
**Rationale**: Instead of a full `[vocab_size, dim]` embedding matrix, use `[vocab_size, k] @ [k, dim]` with k << dim. This dramatically reduces embedding parameter count.

**Example**: With vocab=1024, dim=512, k=64: 1024x64 + 64x512 = 98K params vs 524K (5.3x reduction). Freed bytes fund more transformer layers.

---

## 3. Architecture Innovations

### 3.1 GQA Head Count Optimization
**Current**: Likely 4-8 heads with grouped query attention. Experiment with more aggressive grouping (e.g., 1 KV head per 4 Q heads) to reduce attention parameter count while maintaining quality.

### 3.2 Multi-Scale Feature Mixing
**Rationale**: Add lightweight cross-layer attention or feature pyramid connections beyond current U-Net skip connections. Could improve information flow for minimal parameter cost.

### 3.3 Gated Linear Units (GLU) Variants
**Current**: ReLU² activation. SwiGLU was tried but was too slow.

**Alternative**: Try GEGLU or ReGLU which may have better speed/quality tradeoffs than SwiGLU while being faster.

**Key insight from failed SwiGLU**: The 45% slowdown killed it. Any activation change must be benchmarked for throughput on H100, not just quality.

### 3.4 Shared/Tied Layer Parameters
**Rationale**: Weight-sharing across non-adjacent layers (e.g., layers 1&5 share attention, layers 2&6 share MLP). This provides more effective depth without proportional parameter increase.

**Note**: "Depth recurrence" (using same layer twice) already failed (-0.051 bpb). But partial sharing (share only attention OR MLP, not both) hasn't been tried.

### 3.5 State Space Model (SSM) / Mamba Hybrid
**Rationale**: Replace some attention layers with Mamba/S4 layers. SSMs are O(n) vs O(n²) for attention, enabling longer sequences at lower compute cost.

**Risk**: Custom CUDA kernels needed for efficient Mamba on H100. Implementation complexity within 1500-line limit.

**Promising variant**: Use SSM for early layers (pattern detection) and attention for later layers (in-context learning). Hybrid architectures show strong results in recent literature.

### 3.6 Differential Attention
**Rationale**: Recent work on "Differential Transformer" shows that computing attention as the difference of two softmax attention maps reduces noise and improves signal. Requires 2x attention heads but each can be smaller.

### 3.7 Multi-Token Prediction Head
**Rationale**: Train the model to predict not just the next token but the next 2-4 tokens. This provides a richer training signal per forward pass, potentially converging faster in the 10-minute window.

**Implementation**: Add small prediction heads for positions +2, +3, +4. Auxiliary loss weighted at 0.1-0.3x main loss.

**Expected gain**: Faster convergence (more gradient signal per step), better representations.

---

## 4. Quantization & Compression Frontiers

### 4.1 Optimal Bit Allocation Per Layer
**Rationale**: Not all layers are equally important. Profile per-layer sensitivity to quantization, then allocate bits accordingly:
- Sensitive layers (first, last, bottleneck): Int8 or FP16
- Robust layers (middle): Int4 or Int5
- MLP vs Attention: Different bit-widths (already done, but could be more granular)

### 4.2 Entropy Coding Instead of zstd
**Rationale**: zstd is a general-purpose compressor. For quantized weights with known distribution, arithmetic coding with a learned prior could achieve better compression ratios.

**Implementation**: Fit a histogram to quantized values per-layer, use it as the prior for an arithmetic encoder. Could save 5-15% over zstd-22.

### 4.3 Structured Pruning + Quantization
**Rationale**: SOTA uses 3% magnitude pruning. More aggressive structured pruning (whole rows/columns) could enable:
- Smaller matrices → less to compress
- Regular sparsity patterns compress better than random
- Pruned model runs faster at eval → more time for test-time training

### 4.4 Weight Clustering (k-means Quantization)
**Rationale**: Instead of uniform quantization levels, cluster weights via k-means to find optimal quantization points. With 32 clusters (5 bits), this places quantization levels where the weight density is highest.

### 4.5 Zstd Dictionary Training
**Rationale**: Train a zstd dictionary on quantized weight patterns. zstd supports pre-trained dictionaries that dramatically improve compression of small, structured data. Include the dictionary in the 16MB artifact.

**Expected gain**: 3-8% better compression over default zstd-22 for structured int5/int6 weight data.

### 4.6 Mixed Compression Strategy
**Rationale**: Different layers may compress better with different algorithms. Use zstd for some, lz4 for others, or custom delta encoding for layers with correlated weights.

---

## 5. Training & Optimization

### 5.1 Progressive Layer Freezing
**Rationale**: Freeze early (well-converged) layers partway through training. This:
- Reduces backward pass compute → more steps in 10 min
- Focuses capacity on less-converged later layers
- Stabilizes early representations for better late-training convergence

### 5.2 Gradient Accumulation with Varying Batch Size
**Rationale**: Use smaller batches (more steps) during warmup for faster initial learning, then larger batches during warmdown for stable convergence.

### 5.3 Lion or SOAP Optimizer
**Rationale**: Lion optimizer uses only sign of momentum (no second moment), potentially faster than Adam for embeddings. SOAP optimizer combines Shampoo-style preconditioning with Adam, showing strong results on language modeling.

**Note**: Muon already handles matrix parameters excellently. The opportunity is in improving the optimizer for embeddings and scalar parameters (currently Adam).

### 5.4 Exponential Moving Average (EMA) Instead of SWA
**Rationale**: EMA maintains a running average with decay, updated every step. This captures the full optimization trajectory rather than periodic snapshots. Could be more effective than SWA with less hyperparameter sensitivity.

**Implementation**: `ema_weight = decay * ema_weight + (1-decay) * current_weight` with decay=0.999.

### 5.5 Label Smoothing
**Rationale**: Moderate label smoothing (0.05-0.1) prevents overconfident predictions, produces smoother weight distributions that quantize better. Minimal compute overhead.

### 5.6 Data Ordering / Curriculum Learning
**Rationale**: Present easier examples first (shorter documents, common vocabulary) and harder examples later. This can accelerate early convergence.

**Implementation**: Sort training shards by average token frequency or document length.

### 5.7 Increased Training Data Utilization
**Rationale**: With 8B tokens (80 shards) and ~7400 steps at batch size ~786K tokens, we see ~5.8B tokens in 10 minutes. Consider whether seeing each token exactly once (no repetition) or strategic repetition of high-value data is optimal.

### 5.8 Auxiliary Losses
- **Next-sentence prediction** as secondary objective
- **Embedding similarity** regularization to keep quantization-friendly distributions
- **Layer-wise contrastive loss** to ensure each layer learns different features

---

## 6. Evaluation-Time Improvements

### 6.1 Deeper Test-Time Training (TTT)
**Current**: LoRA TTT was tried with rank-8, single gradient step. Results were modest.

**Improvement**: Use full fine-tuning (not LoRA) on attention parameters only, with 3-5 gradient steps per document. The eval budget is 10 minutes — current TTT uses <10% of this budget.

**Risk**: Overfitting to individual documents. Mitigate with dropout or early stopping per document.

### 6.2 Adaptive Stride for Sliding Window
**Rationale**: Current stride=64 is fixed. Shorter stride (32 or 16) gives more context overlap but costs more compute. Longer stride (128) is faster but lower quality.

**Optimization**: Use stride=32 where eval budget allows (likely possible with 10 min on 8xH100).

### 6.3 Ensemble of SWA Checkpoints
**Rationale**: Instead of averaging weights (SWA), keep top-K checkpoints and ensemble their predictions (average logits). This preserves each model's strengths.

**Cost**: K forward passes instead of 1. With K=3 and 10-min eval budget, feasible if each pass takes ~30s.

### 6.4 Temperature Scaling at Evaluation
**Rationale**: Post-hoc temperature scaling can improve calibration. If the model is overconfident, T>1 softens predictions; if underconfident, T<1 sharpens them. Optimal T found via small validation subset.

### 6.5 Context Distillation at Eval Time
**Rationale**: For each eval document, first process a "summary" prefix that primes the model, then evaluate with this context. The summary could be the document's first few sentences processed through the model to create a compressed context state.

---

## 7. Negative Results (What Failed)

Documented failures from submissions — avoid repeating these:

| Technique | Result | Why It Failed |
|-----------|--------|---------------|
| SwiGLU activation | net negative | 45% slower → fewer steps in 10 min |
| Depth recurrence (layer reuse) | +0.051 bpb | Halved effective steps |
| Late-stage QAT only | marginal | Not enough training iterations for adaptation |
| LZMA compression | worse than zlib | Poor on int8 weight data specifically |
| Higher embedding LR (0.08) | hurt convergence | Embeddings need stability |
| Weight decay on single GPU | no benefit | Too short training for regularization effect |
| QAT int6 late start | overhead not justified | Need full-training QAT or nothing |

---

## 8. Prioritized Roadmap

Ranked by estimated impact/effort ratio:

### Tier 1: High Impact, Moderate Effort
1. **Int4 QAT for MLP weights** — If successful, funds 11th-12th layer. Build on existing Int5 MLP codebase.
2. **Embedding factorization** — Simple change, saves ~400K params for more depth.
3. **Stride=32 sliding window** — Pure eval change, should fit in 10-min budget.
4. **EMA instead of SWA** — Drop-in replacement, potentially smoother weight averaging.
5. **Multi-token prediction** — Richer training signal per step, moderate implementation.

### Tier 2: Medium Impact, Moderate Effort
6. **Per-group quantization** (groups of 32-64) — Finer scale granularity.
7. **Dynamic sequence length curriculum** — Short→long during training.
8. **Weight rotation (Hadamard)** before quantization — Better Int4/Int5 quality.
9. **Progressive layer freezing** — More steps in fixed wall clock.
10. **Label smoothing** — One-line change, may help quantization.

### Tier 3: High Impact, High Effort
11. **Mixture of Experts** — Doubles MLP capacity, complex implementation.
12. **SSM/Mamba hybrid layers** — Needs custom kernels, 1500-line limit is tight.
13. **Entropy coding** (arithmetic coder with learned prior) — Better than zstd, but implementation cost.
14. **Full test-time training** — More aggressive than LoRA TTT, risky overfitting.

### Tier 4: Speculative / Needs Validation
15. **Optimal tokenizer vocab size search** — May be very impactful but requires full pipeline changes.
16. **Knowledge distillation** from longer training run — Rule compliance unclear.
17. **zstd dictionary training** — Needs experimentation to validate gains.
18. **Differential attention** — Novel, unproven in this regime.

---

## Key Insight

The current SOTA at 1.1428 bpb is already heavily optimized. The remaining gains are likely to come from:

1. **Pushing quantization lower** (Int4 for some weights) to fund more parameters
2. **Better compression** to squeeze more model into 16MB
3. **More aggressive eval-time compute** (the 10-min eval budget is underutilized)
4. **Architecture changes** that improve quality per parameter (MoE, factorized embeddings)

The binding constraint analysis suggests:
- **Parameter budget** (16MB) is the primary bottleneck — any technique that improves compression or reduces parameter count per quality unit is high-value
- **Training time** (10 min) is secondary — we complete ~7400 steps which is sufficient for convergence
- **Eval time** (10 min) is significantly underutilized — current eval takes ~90s, leaving 8+ minutes for test-time training or ensemble methods

---

## 9. Insights from Web Research (External Sources)

### 9.1 Looped / Recursive Transformers (HIGH PRIORITY - Untried)
Reuse L unique layers k times for effective depth L*k at L layers of parameter cost. Recent work shows:
- **Loop(L x k)** achieves comparable accuracy to full-depth models at **25-55% of parameter cost** (ICLR 2026)
- **Relaxed Recursive Transformers** (DeepMind): Add layer-specific LoRA modules to shared blocks; 13.5% accuracy improvement
- **Not yet explored in any Parameter Golf submission**
- Could allow effective 20-layer depth with 10 layers of parameters

### 9.2 BitNet b1.58 Ternary Training (SPECULATIVE)
Microsoft's ternary weight {-1, 0, +1} training achieves comparable performance at same parameter count. "BitNet b1.58 Reloaded" demonstrated this works on small networks (100K-48M params) using median-based quantization. At ~1.58 bits/param vs current ~5-6 bits, this could allow 3-4x more parameters in 16MB. Unproven in this regime.

### 9.3 CRVQ (Channel-Relaxed Vector Quantization)
Reduces perplexity by 39% over AQLM with only 0.06-bit overhead by reordering critical weight channels and using extended codebooks. Could outperform current per-row scalar quantization.

### 9.4 Turbo-Muon Optimizer
Achieves 2.8x per-layer speedup via spectral preconditioning of the Newton-Schulz step. Could save wall-clock time, allowing more training steps in 10 minutes.

### 9.5 Gluon Optimizer Framework (ICML 2025)
Unifies Muon and Scion with convergence guarantees. Theoretical stepsizes match empirical fine-tuned values — could reduce hyperparameter search time.

### 9.6 Byte Latent Transformer / ByteFlow Net
- **Byte Latent Transformer** (ACL 2025): Dynamically segments input into variable-length byte patches guided by entropy. Matches Llama 3 at scale.
- **ByteFlow Net** (March 2026): Tokenizer-free, learns self-tokenization from raw bytes. Outperforms BPE models.
- Could eliminate embedding table entirely, but requires significant architecture changes.

### 9.7 TTT Research Updates
- **TLM (ICML 2025)**: 20%+ improvement via perplexity minimization with LoRA and high-perplexity sample selection
- **TTT-E2E (NVIDIA, Jan 2026)**: End-to-end TTT that scales with context length
- Current SOTA (1.1428) does NOT use TTT — combining SOTA arch + TTT is low-hanging fruit

### 9.8 EMA vs SWA (Feb 2025 Research)
Recent paper shows ~1% of training budget is optimal EMA averaging window. "Early Weight Averaging meets High Learning Rates" (LAWA) outperforms standard SWA with spaced checkpoints, especially with high learning rates.

### 9.9 NanoGPT Speedrunning Transferable Techniques
Record dropped from 45 min to 2.86 min. Most techniques already adopted, but **TokenMonster tokenizer** (vocabulary optimization) has not been explored in Parameter Golf.

### Sources
- [ICLR 2026 Looped Transformers](https://openreview.net/pdf/183334103d5fda67d365e08fec721accd09b8ec8.pdf)
- [BitNet b1.58 Reloaded](https://arxiv.org/html/2407.09527v1)
- [CRVQ](https://arxiv.org/html/2412.09282)
- [Turbo-Muon](https://arxiv.org/pdf/2512.04632)
- [Gluon](https://arxiv.org/abs/2505.13416)
- [ByteFlow Net](https://arxiv.org/html/2603.03583)
- [TLM TTT](https://arxiv.org/abs/2505.20633)
- [EMA research](https://arxiv.org/pdf/2502.06761)
- [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt)
