import torch

# Check if MPS (Metal Performance Shaders) is available for Apple Silicon
print("MPS (Apple Silicon) available:", torch.backends.mps.is_available())

# Check if CUDA is available (unlikely on Mac)
print("CUDA available:", torch.cuda.is_available())

# Check current device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)