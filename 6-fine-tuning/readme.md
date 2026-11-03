
# Model Quantization on Apple Silicon (MPS) or CPU

This guide helps you set up your environment and run `dev_model_quantization.py` for quantized language model inference using Hugging Face Transformers and bitsandbytes.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3) recommended for MPS acceleration.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

## Setup Instructions

### 1. Create and Activate Conda Environment

```bash
conda create -n myenv3.12 python=3.12 -y
conda activate myenv3.12
```

### 2. Install packages from `environment.yml`

```bash
conda env update -f environment.yml
```

### 3. (Optional) Check Device Support

```bash
python system_check.py
```

### 5. Run the Quantization Script

```bash
python dev_model_quantization.py
```

## Troubleshooting

- If you see errors related to MPS, ensure you are running on Apple Silicon and have the latest PyTorch.
- The script will automatically use CPU if MPS is unavailable.

## Files

- `system_check.py`: Checks device availability (MPS/CPU).
- `dev_model_quantization.py`: Main script for quantized model loading and inference.


## References

- [PyTorch MPS Docs](https://pytorch.org/docs/stable/notes/mps.html)
- [Transformers Quantization](https://huggingface.co/docs/transformers/main_classes/quantization)

