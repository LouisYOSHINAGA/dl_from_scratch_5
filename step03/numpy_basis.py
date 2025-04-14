import numpy as np

x: np.ndarray = np.array([1, 2, 3])
print(f"{x.__class__=}")
print(f"{x.shape=}")
print(f"{x.ndim=}")

W: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
print(f"{W.ndim=}")
print(f"{W.shape=}")

X: np.ndarray = np.array([[0, 1, 2], [3, 4, 5]])
print(f"{W + X =}")
print(f"{W * X =}")

a: np.ndarray = np.array([1, 2, 3])
b: np.ndarray = np.array([4, 5, 6])
y: float = np.dot(a, b)
print(f"{y=}")

A: np.ndarray = np.array([[1, 2], [3, 4]])
B: np.ndarray = np.array([[5, 6], [7, 8]])
Y: np.ndarray = np.dot(A, B)
print(f"{Y=}")