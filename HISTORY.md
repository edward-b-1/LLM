# Development History

A chronological record of what was built and why things changed.

---

## 1. Initial model

Built the simplest viable decoder-only transformer for causal language modelling:

- `model/attention.py` — single-head causal self-attention using `F.scaled_dot_product_attention(is_causal=True)` (dispatches to Flash Attention automatically)
- `model/feedforward.py` — two-layer MLP, GELU activation, hidden dim = 4 × d_model
- `model/block.py` — Pre-LayerNorm transformer block with residual connections
- `model/gpt.py` — token embeddings + learned absolute positional embeddings, N blocks, LM head with weight tying to token embeddings

Initial config: d_model=256, n_layers=4, n_heads=1.

---

## 2. Data pipeline

- `data/prepare.py` — shared tokenizer (GPT-2 BPE, 50,257 tokens) and `save()` utility
- `data/shakespeare.py` — TinyShakespeare dataset for fast iteration
- `data/dataset.py` — `TokenDataset` using `np.memmap` so the binary token file is never fully loaded into RAM
- `train.py` — training loop with DataLoader, checkpoint save/resume, CLI arguments

---

## 3. Wikipedia data pipeline

Added Wikipedia as a large-scale training dataset. Two memory problems were encountered and fixed:

**Preparation OOM.** Accumulating tokens in a Python list uses ~28 bytes per integer. At ~4B Wikipedia tokens this requires ~112GB RAM. Fixed by switching to `array.array('H')` (2 bytes/token) with incremental flushes to disk every 10M tokens. Peak RAM dropped to ~450MB.

**Training OOM.** `TokenDataset` originally used stride=1, giving ~4.3B samples for Wikipedia. PyTorch's DataLoader pre-generates a shuffled index array before training — 4.3B indices × 8 bytes = 34GB RAM, crashing the machine before a single training step. Fixed by switching to non-overlapping chunks (stride = context_length = 1024), reducing the sample count to ~4.2M.

---

## 4. Optimizer experiments and gradient clipping

Ran SGD → SGD+Momentum → AdamW in sequence on TinyShakespeare to observe the effect of each. AdamW requires a much lower learning rate than SGD (3e-4 rather than 0.1) because its adaptive per-parameter scaling amplifies the effective step size.

Gradient clipping (`clip_grad_norm_`, max_norm=1.0) was added after observing loss spikes during SGD training. It rescales the entire gradient vector when its norm exceeds the threshold, preventing occasional large updates from destabilising training.

---

## 5. Model scaling

Scaled up from d=256 / 4 layers / 1 head to **d=512 / 8 layers / 8 heads** (~25.7M unique parameters). The initial model used a small fraction of the RTX 5090's 32GB VRAM, and small matrix shapes underutilise tensor cores — larger matrices amortise the pipeline startup cost and sustain higher throughput.

---

## 6. Shakespeare Complete dataset

Discovered TinyShakespeare does not contain Hamlet. Added `data/shakespeare_complete.py` to download the complete works from Project Gutenberg with header/footer stripping.

---

## 7. Checkpoint and training management

Checkpointing saves model weights, optimizer state, and both config objects to `checkpoints/`. An optional `--run-name` parameter sets the filename prefix, and `--save-interval` controls how frequently checkpoints are written.

---

## 8. LR scheduling

Added cosine decay with linear warmup. The warmup phase (default: 1,000 steps) prevents large early updates while Adam's gradient variance estimates are still initialising. Cosine decay then drives the LR toward zero over the course of training, allowing the model to converge tightly into a minimum rather than oscillating indefinitely at a fixed LR.

This was motivated by the memorization experiment: a model trained at constant LR failed to recall low-frequency patterns (the document header, seen once) with sufficient confidence. With cosine decay to zero and dropout disabled, the model converged sharply enough to reproduce the training corpus verbatim from position 0 after 200k steps.
