import torch

# 检查是否能使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 获取 GPU 设备信息
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device)}")
    print(f"Memory Cached: {torch.cuda.memory_reserved(device)}")
