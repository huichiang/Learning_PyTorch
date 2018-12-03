from __future__ import print_function
import torch

#Initialise matrices
x1 = torch.empty(5,3)
x2 = torch.rand(5,3)
x3 = torch.zeros(5, 3, dtype=torch.long)
x4 = torch.tensor([[1,2,3],[4,5,6]])
x5 = x1.new_ones(5, 3, dtype=torch.float)  


#Track gradient
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
z = y * y * 3
out = z.mean()
print(z, out)
print(x.grad)
