# RTX 5090 / Blackwell Architecture Notes

Reference notes on the GPU hardware relevant to training decisions.

---

## RTX 5090 Specifications

| Property | Value |
|---|---|
| Architecture | Blackwell (GB202) |
| Compute capability | SM 12.0 |
| SMs enabled | 170 (of 192 on full die — binned for yield/segmentation) |
| VRAM | 32GB GDDR7 |
| Memory bandwidth | ~1.8 TB/s |
| L2 cache | 96MB |

Verify in Python:
```python
import torch
print(torch.cuda.get_device_properties(0))
# SM 12.0, 170 SMs, 32606MB VRAM
```

---

## Inside a Streaming Multiprocessor (SM)

The SM is the fundamental compute unit. The RTX 5090 has 170 of them.

Each SM contains:

| Unit | Count | Purpose |
|---|---|---|
| FP32 CUDA cores | 128 | General arithmetic (one FMA per thread per clock) |
| INT32 cores | 64 | Integer / address calculation |
| Tensor cores | 4 | Matrix multiply-accumulate tiles |
| Special Function Units (SFUs) | 16 | `sin`, `cos`, `exp`, `log`, `rcp`, `rsqrt` |
| Load/Store Units (LSUs) | 64 | Memory transactions |
| Warp schedulers | 4 | Issue instructions to resident warps |
| Register file | ~65,536 × 32-bit | Per-thread fast storage |
| Shared memory / L1 cache | 128KB | Programmer-controlled fast scratchpad |

Across all 170 SMs:
- **21,760 FP32 CUDA cores**
- **680 tensor cores**

---

## Threads, Warps, and Thread Blocks

```
1 thread          — executes one instruction at a time
32 threads        — 1 warp (SIMT: all 32 execute the same instruction simultaneously)
up to 1024 threads — 1 thread block (max 32 warps per block)
up to 2048 threads — max resident per SM (64 warps across ≥2 blocks)
× 170 SMs         — 348,160 threads resident across the whole GPU
```

**SIMT (Single Instruction, Multiple Thread):** all 32 threads in a warp execute the same instruction on different data each cycle. If threads diverge (e.g. an `if` where some threads take each branch), both branches execute sequentially with threads masked off — this is warp divergence and wastes cycles. Deep learning kernels are written to avoid it.

**Latency hiding:** an SM keeps up to 64 warps resident simultaneously. With 4 warp schedulers issuing 4 instructions per clock, while warp A waits ~400 cycles for VRAM data, warps B/C/D/... keep the compute units busy. High warp occupancy is essential for performance.

**Occupancy:** ratio of resident warps to the 64-warp maximum per SM. Limited by register file usage and shared memory allocation per block. Flash Attention accepts lower occupancy in exchange for dramatically reduced VRAM traffic.

---

## CUDA Cores vs Tensor Cores

**CUDA cores** perform one FMA (Fused Multiply-Add: `result = A × B + C`) per thread per clock. With 128 FP32 cores per SM: 128 FMAs/clock/SM.

**Tensor cores** perform a matrix-tile FMA: `D = A × B + C` where A, B, C, D are small matrices. All 32 warp threads participate cooperatively — each thread holds a fragment of the matrix in its registers. The operation takes ~16–32 cycles (pipelined), not one.

For BF16 on a 16×16×16 tile: 4,096 multiply-accumulates per tensor core per operation. Four tensor cores per SM: 16,384 MACs/SM — approximately **128× the throughput** of CUDA cores for matrix multiplies.

Tensor cores are the only unit that can perform matrix multiply-accumulate. Everything else (residual addition, softmax, LayerNorm, GELU) runs on CUDA cores and SFUs.

---

## Precision Formats and Tensor Core Support

RTX Blackwell tensor cores support (confirmed from NVIDIA Blackwell architecture whitepaper):

| Format | Bits | Tensor Core | Notes |
|---|---|---|---|
| FP4 | 4 | Yes | New in RTX Blackwell — inference/quantisation |
| FP6 | 6 | Yes | New in RTX Blackwell |
| FP8 | 8 | Yes | Inherited from Hopper — aggressive training |
| INT8 | 8 | Yes | Integer operations |
| FP16 | 16 | Yes | Standard half-precision |
| BF16 | 16 | Yes | Preferred for training (FP32-equivalent range) |
| TF32 | 19 | Yes | Transparent FP32 acceleration (PyTorch default) |
| FP32 | 32 | Via TF32 | Full precision |
| FP64 | 64 | Yes (limited) | Scientific computing |

**TF32** is not a storage format — it's a compute mode. When PyTorch runs an FP32 matrix multiply, it internally truncates the mantissa to 10 bits, runs the multiply in tensor cores, and returns FP32. Transparent to the user, enabled by default.

### Choosing precision

| Question | If yes → use |
|---|---|
| Does this value survive many gradient steps? | FP32 (optimizer states) |
| Can this value exceed ~65,000? | BF16 or FP32 (FP16 overflows here) |
| Is this a matrix multiply on the forward pass? | BF16 (tensor cores) |

**Standard mixed-precision training recipe:**

| What | Precision |
|---|---|
| Forward pass activations | BF16 |
| Gradients | BF16 (accumulated in FP32) |
| Optimizer states (Adam m, v) | FP32 |
| Master weight copy | FP32 |

`torch.autocast(device_type='cuda', dtype=torch.bfloat16)` handles this automatically.

---

## Floating Point Format Internals

All floating point formats share the same structure:

```
[ sign | exponent | mantissa ]
```

| Format | Sign | Exponent | Mantissa | Max value |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | ~3×10³⁸ |
| BF16 | 1 | 8 | 7 | ~3×10³⁸ |
| FP16 | 1 | 5 | 10 | ~65,504 |
| TF32 | 1 | 8 | 10 | ~3×10³⁸ |

BF16 keeps the same 8 exponent bits as FP32 (same range), sacrificing mantissa precision. FP16 has only 5 exponent bits — a maximum of ~65,504 — making it prone to overflow during training without loss scaling. BF16 avoids this entirely, which is why it replaced FP16 as the standard training format.

---

## Memory Hierarchy

```
Registers          ~0 cycles      Per-thread, private (~65K × 32-bit per SM)
Shared Memory      ~5 cycles      128KB per SM, shared within a thread block
L1 Cache           ~5 cycles      Same physical 128KB, automatic
L2 Cache           ~30 cycles     96MB, shared across all 170 SMs
VRAM (GDDR7)       ~400 cycles    32GB, ~1.8 TB/s bandwidth
System RAM         ~2000 cycles   Via PCIe
```

GPU kernel optimisation is fundamentally about keeping data as high in this hierarchy as possible. Flash Attention exploits this by tiling Q, K, V to fit in the 128KB shared memory, computing the full attention output without ever writing the T×T attention matrix to VRAM.

---

## Tensor Core Internals — Systolic Array

Tensor cores are believed to implement a **systolic array** — a grid of simple multiply-add cells where data flows rhythmically through the grid (A values flow right, B values flow down, partial sums accumulate at each cell). No cell needs visibility of the whole matrix.

For a 3×3 example, the array fills over `2N-1 = 5` cycles, with only 1 cycle at full utilisation. However, real matrix multiplies consist of thousands of tiles. Once the pipeline is full:

```
efficiency ≈ (N_tiles - 1) / N_tiles  →  ~100% for large matrices
```

This is why matrix shapes matter — small matrices (from small batch sizes or small `d_model`) don't have enough tiles to amortise the pipeline startup cost, leaving most tensor core capacity idle.

---

## wgmma Instruction Variants

`wgmma` (warpgroup matrix multiply-accumulate) is the Hopper/Blackwell tensor core instruction family. It comes in variants across:

- **Input precision:** FP4, FP6, FP8, FP16, BF16, TF32
- **Accumulator:** always FP32 internally (output can be cast)
- **Tile shapes:** vary by precision (k dimension doubles as bit-width halves)
- **Transposition flags:** A and B can each be independently transposed

| Precision | Typical tile (m×n×k) |
|---|---|
| FP64 | 8×8×4 |
| TF32 | 16×16×8 |
| FP16 / BF16 | 16×16×16 |
| FP8 | 16×16×32 |
| FP4 | 16×16×64 |

The k dimension doubles as precision halves because lower-precision values pack into the same register space, allowing the tensor core to process twice as many elements per cycle.

---

## Benchmark Results

Sustained matmul throughput on 4096×4096 matrices (measured):

| Format | Achieved TFLOPS |
|---|---|
| FP32 (via TF32) | 61.8 |
| BF16 | 180.9 |
| FP16 | 203.7 |

Gap from theoretical peak is due to memory bandwidth overhead, cuBLAS kernel selection on a new architecture, and absence of structured sparsity. Expect improvements as drivers and cuBLAS mature for SM 12.0.

Benchmark script:
```python
import torch, time

def benchmark(dtype, size=4096, iters=200):
    a = torch.randn(size, size, device='cuda', dtype=dtype)
    b = torch.randn(size, size, device='cuda', dtype=dtype)
    for _ in range(10):           # warmup
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    flops = 2 * size**3 * iters
    return flops / (time.perf_counter() - t0) / 1e12

for dtype, name in [(torch.float32, "FP32"), (torch.bfloat16, "BF16"), (torch.float16, "FP16")]:
    print(f"{name}: {benchmark(dtype):.1f} TFLOPS")
```

---

## Key Papers

| Paper | Relevance |
|---|---|
| Rumelhart, Hinton & Williams (1986) — *Learning representations by back-propagating errors* | Backpropagation |
| Vaswani et al. (2017) — *Attention Is All You Need* | Original transformer |
| He et al. (2015) — *Deep Residual Learning for Image Recognition* | Residual connections |
