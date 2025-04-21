import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


num: np.ndarray = np.array([
    [ 2.0, 54.50],
    [ 4.3, 80.00]
])
covs: np.ndarray = np.array([
    [[ 0.07,  0.44],
     [ 0.44, 33.70]],
    [[ 0.17,  0.94],
     [ 0.94, 36.00]]
])
phis: np.ndarray = np.array([0.35, 0.65])


def multivariate_normal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    det: float = np.linalg.det(cov)
    inv: np.ndarray = np.linalg.inv(cov)
    z: float = 1 / np.sqrt((2 * np.pi) ** len(x) * det)
    y: float = z * np.exp(- (x - mu).T @ inv @ (x - mu) / 2.0)  # type: ignore
    return y

def gmm(x: np.ndarray, phis: np.ndarray, mus: np.ndarray, covs: np.ndarray) -> float:
    y: float = 0
    for k in range(len(x)):
        phi: float = phis[k]
        mu: np.ndarray = mus[k]
        cov: np.ndarray = covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


xs: np.ndarray = np.arange(1, 6, 0.1)
ys: np.ndarray = np.arange(40, 100, 0.1)
X, Y = np.meshgrid(xs, ys)
Z: np.ndarray = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x: np.ndarray = np.array([X[i, j], Y[i, j]])
        Z[i, j] = gmm(x, phis, num, covs)


# fig. 4-9
fig: Figure = plt.figure(figsize=(8, 4))
ax1: Axes3D = fig.add_subplot(1, 2, 1, projection="3d")  # type: ignore
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.plot_surface(X, Y, Z, cmap="viridis")

ax2: Axes = fig.add_subplot(1, 2, 2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.contour(X, Y, Z)
plt.tight_layout()
plt.show()