import numpy as np

A: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
print(f"{A=}")
print(f"{A.T=}")

A: np.ndarray = np.array([[3, 4], [5, 6]])
d: float = np.linalg.det(A)
B: np.ndarray = np.linalg.inv(A)
print(f"{d=}")
print(f"{B=}")
print(f"{A @ B =}")

def multivariate_normal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.float64:
    det: np.float64 = np.linalg.det(cov)
    inv: np.ndarray = np.linalg.inv(cov)
    D: int = len(x)
    z: np.float64 = 1 / np.sqrt((2 * np.pi) ** D * det)
    y: np.float64 = z * np.exp(- (x - mu).T @ inv @ (x - mu) / 2.0)  # type: ignore
    return y

x: np.ndarray = np.array([0, 0])
mu: np.ndarray = np.array([1, 2])
cov: np.ndarray = np.array([[1, 0], [0, 1]])
y: np.float64 = multivariate_normal(x, mu, cov)
print(f"{y=}")