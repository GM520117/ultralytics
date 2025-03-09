import torch
import torch_directml

print("Torch version:", torch.__version__)
print("Torch DirectML available:", torch_directml.is_available())
