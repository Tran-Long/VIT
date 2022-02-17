import torch

x = torch.randn((1, 2, 3, 4))
y = torch.randn((1, 2, 3, 4))
y_t = torch.transpose(y, -2, -1)
print(torch.matmul(x, y_t).shape)