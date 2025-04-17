import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def multivariate_normal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.float64:
    det: np.float64 = np.linalg.det(cov)
    inv: np.ndarray = np.linalg.inv(cov)
    D: int = len(x)
    z: np.float64 = 1 / np.sqrt((2 * np.pi) ** D * det)
    y: np.float64 = z * np.exp(- (x - mu).T @ inv @ (x - mu) / 2.0)  # type: ignore
    return y


mu: np.ndarray = np.array([0.5, -0.2])
cov: np.ndarray = np.array([[2.0, 0.3], [0.3, 0.5]])

xs: np.ndarray = np.arange(-5, 5, 0.1)
ys: np.ndarray = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(xs, ys)
Z: np.ndarray = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x: np.ndarray = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)


# fig. 3-13
fig: Figure = plt.figure(figsize=(8, 4))
ax1: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")  # type: ignore
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.plot_surface(X, Y, Z, cmap="viridis")

ax2: Axes3D = fig.add_subplot(1, 2, 2)  # type: ignore
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.contour(X, Y, Z)
plt.tight_layout()
plt.show()