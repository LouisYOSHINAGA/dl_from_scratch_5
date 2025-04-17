import os
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


path: str = os.path.join(os.path.dirname(__file__), "height_weight.txt")
xs: np.ndarray = np.loadtxt(path)

mu: np.ndarray = np.mean(xs, axis=0)
cov: np.ndarray = np.cov(xs, rowvar=False)

small_xs: np.ndarray = xs[:500]
X, Y = np.meshgrid(np.arange(150, 195, 0.5), np.arange(45, 75, 0.5))
Z: np.ndarray = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x: np.ndarray = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)


# fig. 3-17
fig: Figure = plt.figure(figsize=(8, 4))
ax1: Axes3D = fig.add_subplot(1, 2, 1, projection='3d')  # type: ignore
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

ax2: Axes3D = fig.add_subplot(1, 2, 2)  # type: ignore
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(156, 189)
ax2.set_ylim(36, 79)
ax2.scatter(small_xs[:,0], small_xs[:,1])
ax2.contour(X, Y, Z)
plt.tight_layout()
plt.show()