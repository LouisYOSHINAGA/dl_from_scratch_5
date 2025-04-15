import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D


X: np.ndarray = np.array([
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 1, 2],
    [-2, -1, 0, 1, 2],
])
Y: np.ndarray = np.array([
    [-2, -2, -2, -2, -2],
    [-1, -1, -1, -1, -1],
    [ 0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1],
    [ 2,  2,  2,  2,  2],
])
Z: np.ndarray = X ** 2 + Y ** 2

# fig. 3-10
ax0: Axes3D = plt.axes(projection="3d")  # type: ignore
ax0.plot_surface(X, Y, Z, cmap="viridis")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.set_zlabel("z")
plt.show()


xs: np.ndarray = np.arange(-2, 2, 0.1)
ys: np.ndarray = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = X ** 2 + Y ** 2

# fig. 3-11
ax1: Axes3D = plt.axes(projection="3d")  # type: ignore
ax1.plot_surface(X, Y, Z, cmap="viridis")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
plt.show()


x: np.ndarray = np.arange(-2, 2, 0.1)
y: np.ndarray = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(x, y)
Z: np.ndarray = X ** 2 + Y ** 2

# fig. 3-12
ax2: axes.Axes = plt.axes()
ax2.contour(X, Y, Z)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.show()