import torch

print("PyTorch version:", torch.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(3, 3, device=device)
print("A random tensor living on", x.device)
print(x @ x)  # matrix multiply, run on the GPU
