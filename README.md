# LLM From Scratch

A transformer-based language model built from scratch in PyTorch, loosely following Raschka's *Build a Large Language Model (From Scratch)* but not strictly. The goal is to understand every design decision by starting with the simplest possible architecture and expanding incrementally — observing the effect of each change.

---

## Environment

- **GPU:** NVIDIA RTX 5090 (Blackwell GB202, SM 12.0, 170 SMs enabled, 32GB VRAM)
- **OS:** Windows 11 + WSL2 (Ubuntu 24.04)
- **PyTorch:** Nightly build with CUDA 12.8 (required for SM 12.0 support)
- **All code and data:** WSL2 ext4 filesystem (not `/mnt/c/`) for I/O performance

### Verifying the environment

```python
import torch
print(torch.cuda.is_available())           # True
print(torch.cuda.get_device_name(0))       # NVIDIA GeForce RTX 5090
print(torch.cuda.get_device_capability())  # (12, 0)
```

### Installing PyTorch nightly

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Project Structure

```
llm/
├── config.py                  Model hyperparameters (ModelConfig dataclass)
├── train.py                   Training loop (TrainConfig dataclass)
├── model/
│   ├── attention.py           CausalSelfAttention
│   ├── feedforward.py         FFN (two-layer MLP with GELU)
│   ├── block.py               TransformerBlock (attention + FFN + residuals + LayerNorm)
│   └── gpt.py                 GPT — full model, embeddings, loss
├── data/
│   ├── prepare.py             Shared utilities: tokenizer, save(), paths
│   ├── dataset.py             TokenDataset — PyTorch Dataset using np.memmap
│   ├── shakespeare.py         TinyShakespeare preparation
│   ├── wikipedia.py           Wikipedia preparation (memory-efficient, single-process)
│   ├── wikipedia_parallel.py  Wikipedia preparation (16-process parallel)
│   ├── README.md              Data pipeline documentation
│   └── INSPECTION.md          IPython snippets for inspecting raw and tokenised data
├── datasets/                  Binary token files — gitignored
├── checkpoints/               Saved model checkpoints — gitignored
├── README.md                  This file
└── HARDWARE.md                RTX 5090 / Blackwell architecture notes
```

---

## Architecture

### Current (simple baseline)

The deliberately minimal starting point. Each component will be swapped out individually so the effect of each change can be observed.

| Component | Current choice | Planned upgrade |
|---|---|---|
| Positional encoding | Learned absolute embeddings | RoPE |
| Attention heads | 1 | 4+ |
| Attention kernel | `F.scaled_dot_product_attention` | Flash Attention 3 |
| Normalisation | Pre-LayerNorm | RMSNorm |
| FFN activation | GELU | SwiGLU |
| Tokenizer | GPT-2 BPE (HuggingFace) | — |

### Hyperparameters (`config.py`)

```python
@dataclass
class ModelConfig:
    vocab_size:     int   = 50257   # GPT-2 BPE vocabulary
    context_length: int   = 1024    # fixed sequence length
    d_model:        int   = 256     # embedding / hidden dimension
    n_heads:        int   = 1       # attention heads
    n_layers:       int   = 4       # transformer blocks
    ffn_mult:       int   = 4       # FFN hidden = ffn_mult * d_model
    dropout:        float = 0.1
```

**Parameter count:** ~16.3M total (~3.4M excluding the shared token embedding).

### Data flow

```
idx (B, T)                    — integer token IDs
  ↓  tok_emb + pos_emb
x (B, T, 256)                 — embedded + positioned, dropout applied
  ↓  × 4 TransformerBlocks
     each block:
       x = x + Attention(LayerNorm(x))
       x = x + FFN(LayerNorm(x))
x (B, T, 256)                 — contextual representations
  ↓  final LayerNorm
  ↓  lm_head (weight-tied to tok_emb)
logits (B, T, 50257)          — token probability scores
  ↓  cross_entropy with targets
L (scalar)                    — causal language modelling loss
```

### Key design decisions

**Pre-LayerNorm** — normalise before each sublayer rather than after. More stable training than the original Post-LN from "Attention Is All You Need". Gradients flow through the residual stream without passing through a normalisation layer.

**Weight tying** — `lm_head` and `tok_emb` share the same weight matrix. Saves 12.9M parameters and enforces geometric consistency: the same space that represents input tokens also scores output predictions.

**Residual connections** — `x = x + sublayer(x)` throughout. The gradient of this with respect to `x` is `∂F/∂x + 1` — the `+1` ensures gradient signal always flows back to early layers regardless of what the sublayer contributes, preventing vanishing gradients.

**`F.scaled_dot_product_attention(is_causal=True)`** — PyTorch's built-in attention operator. Dispatches to Flash Attention kernels automatically on supported hardware. Applies the causal mask without materialising the full T×T matrix in VRAM.

**Causal LM objective** — at each position t, predict token t+1. Input/target pair for a chunk of 1025 tokens: `x = tokens[0:1024]`, `y = tokens[1:1025]`.

---

## Dataset Preparation

Uses the GPT-2 BPE tokenizer (50,257 token vocabulary). All datasets are saved as raw binary files of `uint16` token IDs (2 bytes each). Articles are separated by the EOS token (ID 50256).

```bash
# TinyShakespeare — fast sanity-check dataset (seconds)
python -m data.shakespeare

# Wikipedia — full training dataset, memory-efficient single process
python -m data.wikipedia

# Wikipedia — 16-process parallel version (faster)
python -m data.wikipedia_parallel
```

Output written to `datasets/`. See `data/README.md` for format details and `data/INSPECTION.md` for IPython inspection snippets.

### Memory efficiency

The naive approach of accumulating tokens in a Python list uses ~28 bytes per token. At ~4 billion Wikipedia tokens this requires ~112GB — guaranteed OOM. Our implementation uses:

- `array.array('H')` — 2 bytes per token, flushed to disk every 10M tokens (~20MB peak buffer)
- Generator iteration — no bulk text loaded into RAM upfront
- Raw file I/O for train/val split — no full dataset array in memory at any point

Peak RAM during Wikipedia preparation: ~450MB.

---

## Training

```bash
python train.py
```

### Optimiser progression

Optimisers are run in sequence to observe the effect of each change:

| Stage | Optimiser | LR |
|---|---|---|
| 1 | SGD | 0.1 |
| 2 | SGD + Momentum (0.9) | 0.1 |
| 3 | AdamW | 3e-4 |

Switch by changing one field in `TrainConfig`:

```python
TrainConfig(optimiser="sgd")
TrainConfig(optimiser="sgd_momentum")
TrainConfig(optimiser="adamw")
```

Checkpoints are saved to `checkpoints/` as `.pt` files containing model weights, optimiser state, and both config objects.

---

## Benchmark Results (RTX 5090)

Sustained matmul throughput on 4096×4096 matrices:

| Format | TFLOPS |
|---|---|
| FP32 (via TF32 tensor cores) | 61.8 |
| BF16 | 180.9 |
| FP16 | 203.7 |

---

## Planned Experiments (in order)

1. SGD → SGD+Momentum → AdamW comparison on Shakespeare
2. 1 attention head → 4 heads
3. Learned positional embeddings → RoPE
4. GELU → SwiGLU
5. LayerNorm → RMSNorm
6. Scale `d_model` and `n_layers` up toward 1B parameters
7. Switch training data to Wikipedia
8. BF16 mixed precision
9. Gradient clipping and LR scheduling (warmup + cosine decay)
