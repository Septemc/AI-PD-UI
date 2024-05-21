import torch

# 创建一个形状为 1*2 的张量
tensor = torch.randn(1, 2)

# 对张量进行sigmoid转换
sigmoid_tensor = torch.sigmoid(tensor)
res = torch.max(sigmoid_tensor).item() * 100
res = round(res, 3)
# 打印转换后的张量
print(res)
