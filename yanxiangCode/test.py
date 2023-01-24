import torch
a=torch.arange(1200).view(3,20,20)
b=a.view(20,20,3)
print(a[0])
print(b[...,0])