import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
z= z.mean()
print(z)

v = torch.tensor([0.1, 0.1, 0.001], dtype=torch.float32)
z.backward() #dz/dx
print(x.grad)

#no grad use / and stop it from creating the grad function and the history graph
x.requires_grad_(False)
print(x)

u = x.detach()
print(u)

with torch.no_grad():
    b = x + 2
    print(b)