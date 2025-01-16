import torch
import torch.nn as nn

# 检查并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建模型并将其移动到 GPU
model = nn.Linear(10, 10).to(device)  # 使用 .to(device) 将模型移动到 GPU

# 创建随机数据并将其移动到 GPU
data = torch.randn(64, 10).to(device)  # 同样，将数据移到 GPU

# 进行前向计算
output = model(data)
print(f"Output shape: {output.shape}")
