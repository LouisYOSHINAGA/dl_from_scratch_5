import torch as t
import numpy as np
import matplotlib.pyplot as plt


lr: float = 0.001
iters: int = 10000


def rosenblock(x0: t.Tensor, x1: t.Tensor) -> t.Tensor:
    y: t.Tensor = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0: t.Tensor = t.tensor(0.0, requires_grad=True)
x1: t.Tensor = t.tensor(2.0, requires_grad=True)

for i in range(iters):
    if i % 1000 == 0:
        print(f"{x0.item()=:.5f}, {x1.item()=:.5f}")

    y: t.Tensor = rosenblock(x0, x1)
    y.backward()

    x0.data -= lr * x0.grad.data  # type: ignore
    x1.data -= lr * x1.grad.data  # type: ignore

    x0.grad.zero_()  # type: ignore
    x1.grad.zero_()  # type: ignore

print(f"{x0.item()=:.5f}, {x1.item()=:.5f}")