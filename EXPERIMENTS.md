# Experiments Log

A record of what was built, what was tried, and what the results were.

---

## Model Architecture

Built a decoder-only transformer (GPT-style) from scratch in PyTorch. Components implemented in order:

| File | What it implements |
|---|---|
| `model/attention.py` | Causal self-attention with fused QKV projection, `F.scaled_dot_product_attention(is_causal=True)` |
| `model/feedforward.py` | Two-layer MLP with GELU activation, hidden dim = `ffn_mult × d_model` |
| `model/block.py` | Transformer block: Pre-LayerNorm, attention + FFN, residual connections |
| `model/gpt.py` | Full model: token + positional embeddings, N blocks, final LayerNorm, LM head |

**Key design decisions:**
- **Pre-LayerNorm** (normalise before each sublayer, not after) — more stable gradients than the original "Attention Is All You Need" formulation
- **Weight tying** — `lm_head` shares weights with `tok_emb`, saving ~25M parameters and enforcing that input and output token spaces are geometrically consistent
- **Residual connections** — gradient of `x = x + f(x)` with respect to `x` is `∂f/∂x + 1`; the `+1` term ensures gradients reach early layers regardless of what the sublayer contributes
- **`F.scaled_dot_product_attention`** — dispatches to Flash Attention kernels automatically; applies causal mask without materialising the T×T attention matrix in VRAM
- **BF16 mixed precision** via `torch.autocast` — forward pass and gradients in BF16, optimizer states in FP32; BF16 preferred over FP16 because it has the same exponent range as FP32 (~3×10³⁸ vs ~65,504) so it doesn't overflow during training

**Final model config** (`config.py`):
```
vocab_size:     50257   (GPT-2 BPE)
context_length: 1024
d_model:        512
n_heads:        8       (head_dim = 64)
n_layers:       8
ffn_mult:       4
dropout:        0.1
```
~25.7M unique parameters (51.4M total, ~25.7M subtracted for shared lm_head weight).

---

## Data Pipeline

**Tokenizer:** GPT-2 BPE (HuggingFace `GPT2TokenizerFast`), vocab size 50,257. Saved locally to `tokenizer/` to avoid repeated HuggingFace API calls.

**Storage format:** raw binary `uint16` files (2 bytes/token). `TokenDataset` reads via `np.memmap` — the file is never fully loaded into RAM.

**Datasets prepared:**

| Dataset | Tokens | Notes |
|---|---|---|
| TinyShakespeare | ~1M | Fast sanity-check; does not contain Hamlet |
| Shakespeare Complete | ~1.67M | Project Gutenberg #100, complete works |
| Wikipedia (EN) | ~4.3B | HuggingFace `wikipedia` dataset, 20231101.en |

**Wikipedia memory efficiency problem:** The naive implementation accumulated tokens in a Python list (~28 bytes/int). At 4.3B tokens this requires ~112GB RAM — guaranteed OOM. Fixed with `array.array('H')` (2 bytes/token) flushed to disk every 10M tokens. Peak RAM during preparation: ~450MB.

**DataLoader OOM during training:** `TokenDataset.__len__` originally returned every possible starting position (stride=1). For Wikipedia's 4.3B tokens this produces 4.3B samples; PyTorch's DataLoader pre-generates a shuffled index array of 4.3B × 8 bytes = **34GB RAM** before training starts. Fixed by switching to non-overlapping chunks (stride = `context_length`), reducing sample count to ~4.2M and the index array to ~34MB.

---

## Experiment 1 — Optimizer Comparison (TinyShakespeare)

**Objective:** Observe the effect of SGD → SGD+Momentum → AdamW on the same dataset and model.

**Model:** Small baseline (d=256, 4 layers, 1 head, ~3.4M unique params)  
**Dataset:** TinyShakespeare (~1M tokens)

| Optimizer | LR | Observations |
|---|---|---|
| SGD | 0.1 | Converged slowly, loss spikes observed around steps 340–360 |
| SGD + Momentum (0.9) | 0.1 | Faster convergence than plain SGD |
| AdamW | 0.1 | **Loss exploded to 54.8** — LR too high for AdamW |
| AdamW | 3e-4 | Correct LR; converged to train loss ~0.69, val loss ~7.44 |

**Learning:** AdamW's effective per-parameter step size is `lr / sqrt(v)` where `v` is the running variance of gradients. With `lr=0.1`, the actual steps were ~100× larger than intended for well-scaled parameters. Correct LR for AdamW is typically 1e-4 to 3e-4.

**Gradient clipping** (`clip_grad_norm_`, max_norm=1.0) added after observing loss spikes — rescales the gradient vector if its norm exceeds the threshold, preventing large updates from destabilising training.

---

## Experiment 2 — Model Scaling

**Objective:** Scale the model up to use more of the available 32GB VRAM.

Scaled from d=256 / 4 layers / 1 head to **d=512 / 8 layers / 8 heads**.

- Unique parameter count: ~3.4M → ~25.7M
- VRAM usage at batch_size=16: ~9.5GB
- Initial loss at random init: ~10.93 (close to theoretical `ln(50257) ≈ 10.82` — confirms correct initialisation)
- GPU utilisation improved significantly with larger matrix shapes, since small matrices don't fully amortise tensor core pipeline startup costs

---

## Experiment 3 — Overfitting on Shakespeare Complete

**Objective:** Train to convergence on Shakespeare Complete to study memorization behaviour.

**Dataset:** Shakespeare Complete (~1.67M tokens)  
**Optimizer:** AdamW, lr=3e-4  
**Steps:** ~44,000 (first run), ~200,000 (second run)

**Results at ~44k steps:**
```
train loss: 0.066
val loss:   8.65
```

A train/val gap this large is textbook memorization. The model has seen each training token ~420 times (44k steps × 16 batch × 1024 context ÷ 1.67M tokens). A 25M parameter model will memorize a 1.67M token corpus given sufficient training.

**Cross-entropy loss of 0.066** = average perplexity of e^0.066 ≈ 1.07 — the model is highly confident on training data but has learned nothing transferable.

**Lesson:** There is no point continuing to train on a corpus the model has already memorized. The right move is a much larger dataset (Wikipedia).

---

## Experiment 4 — Memorization Verification

**Objective:** Confirm the model has genuinely memorized the training data by reproducing it verbatim.

**Hypothesis:** If the model has memorized the data, feeding the exact first tokens of the training corpus (at position 0) at temperature=0 should reproduce the training data.

**First attempt — failed:**
- Prompt: `"To be, or not to be"` → model produced plausible Shakespeare but not the Hamlet continuation
- **Reason:** The model was trained on TinyShakespeare, which does not contain Hamlet. Grep confirmed: `cat shakespeare.txt | grep -i 'to be, or'` returned no results.

**Second attempt — failed:**
- Switched to Shakespeare Complete dataset, but preparation script had a bug: `text.find("THE COMPLETE WORKS OF WILLIAM SHAKESPEARE")` used wrong case — actual file has `"The Complete Works of William Shakespeare"`. The `find()` returned -1, the header-stripping was silently skipped, and the model trained on the full Gutenberg file including the boilerplate header.
- Additionally, the footer search `"End of the Project Gutenberg"` also failed (actual: `"*** END OF THE PROJECT GUTENBERG EBOOK 100 ***"`), so the footer was also included.

**Third attempt — failed at 44k steps:**
- Dataset bug fixed. Trained to 44k steps.
- Prompt: `"THE COMPLETE WORKS OF WILLIAM SHAKESPEARE"` (wrong case) → model produced "GLOUCESTER." instead of the expected continuation.
- **Reason 1:** The actual text begins with `"The Complete Works of William Shakespeare"` (mixed case), not all-caps. Case mismatch means the prompt doesn't match what the model saw at position 0.
- **Reason 2:** Even with the right prompt, 44k steps with constant lr=3e-4 was insufficient convergence for low-frequency patterns (the header appears once; character dialogue appears thousands of times).

**Fourth attempt — succeeded at 200k steps:**
- Added cosine LR decay (lr=3e-4 → 0), set dropout=0
- Prompt: `"The Complete Works of William Shakespeare"` at temperature=0 (greedy)
- **Result: model reproduced the training data verbatim from position 0** ✓

**Key insight — position-dependence of absolute embeddings:** A model with learned absolute positional embeddings assigns a different embedding vector to each position (0, 1, ..., 1023). During training, `"To be, or not to be"` appeared in the middle of context windows (at positions ~500–1020), not at position 0. At inference time with a short prompt, those tokens are at positions 0–6. The model has never been trained to predict from this position configuration, so even a memorized model fails. This is one of the core motivations for **RoPE** (Rotary Position Embeddings), which encodes *relative* distance between tokens inside the attention scores rather than absolute position in the embedding.

**Additional insight — autoregressive error cascading:** Even if the first token is predicted slightly wrong (say 80% confidence in the correct token), any single wrong prediction puts the model on a trajectory that diverges entirely from the training data. Subsequent context is now something the model has never seen.

---

## Infrastructure Built

Beyond the model itself, the following training infrastructure was implemented:

- **Checkpoint save/resume** — auto-detects latest checkpoint per run name, resumes from the correct step
- **`--run-name` CLI flag** — checkpoint filename prefix, allowing multiple runs with the same optimizer on different datasets without collisions
- **Cosine LR schedule with linear warmup** — `get_lr(step, cfg)` implements warmup over `warmup_steps` then cosine decay from `lr` to `lr_min`
- **Gradient clipping** — `clip_grad_norm_` with configurable max norm
- **Decoupled save/eval intervals** — evaluation (val loss) and checkpoint saving are now independent; default saves every 10k steps
- **`generate.py`** — temperature sampling with top-k filtering; temperature=0 now correctly performs greedy (argmax) decoding rather than dividing by zero

---

## Planned Next Steps

1. **Wikipedia training** — switch to the 4.3B token Wikipedia corpus; the Shakespeare corpus is too small for this model size
2. **RoPE positional embeddings** — replace learned absolute embeddings; fixes position-dependence at inference time and enables length generalisation
3. **SwiGLU activation** — replace GELU in FFN (used by LLaMA, Gemma, Mistral)
4. **RMSNorm** — replace LayerNorm (simpler, slightly faster)
5. **Scale toward GPT-2 small** — d=768, 12 layers, 12 heads after Wikipedia training validates the pipeline
