import torch as t

x: t.Tensor = t.tensor(5.0, requires_grad=True)
y: t.Tensor = 3 * x ** 2

y.backward()
print(f"{x.grad=}")