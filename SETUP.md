# Setup

## Requirements

- Python 3.10+
- Git

---

## CPU-only (inference only)

Suitable for running `generate.py` on a laptop or any machine without a GPU.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

---

## GPU (NVIDIA CUDA)

Required for training. The correct CUDA version depends on your driver.

**CUDA 12.8 (RTX 5090 / Blackwell):**
```bash
python -m venv venv
source venv/bin/activate

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements-cpu.txt
```

**CUDA 12.4 (most other modern NVIDIA GPUs):**
```bash
python -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-cpu.txt
```

To check which CUDA version your driver supports:
```bash
nvidia-smi
```
The top-right corner of the output shows the maximum supported CUDA version.

---

## Verifying the installation

```python
import torch
print(torch.cuda.is_available())       # False on CPU-only, True with GPU
print(torch.cuda.get_device_name(0))   # GPU name, if available
```

---

## Running inference

Copy the following to the target machine:

- The codebase (this directory)
- The checkpoint file (`checkpoints/*.pt`)

Then:

```bash
python generate.py --checkpoint checkpoints/<checkpoint>.pt \
    --prompt "Question: What is the capital of France?\nAnswer:" \
    --temperature 0.0
```
