import torch
from torch import nn
from tqdm import trange

x = torch.tensor([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1],]).float()
y = torch.tensor([0, 1, 1, 0]).float()

f1 = nn.Linear(2, 2)
act = nn.Sigmoid()
f2 = nn.Linear(2, 1)
params = list(f1.parameters()) + list(f2.parameters())
optim = torch.optim.Adam(params)

for _ in trange(10000):
    y_hat = f2(act(f1(x))).squeeze()
    error = y - y_hat
    loss = error.t().dot(error)
    optim.zero_grad()
    loss.backward()
    optim.step()

print()
print(y_hat)
print(f1.weight)
print(f1.bias)
print(loss.item())
