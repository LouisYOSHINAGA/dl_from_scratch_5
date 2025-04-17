import os
import numpy as np
import matplotlib.pyplot as plt

path: str = os.path.join(os.path.dirname(__file__), "height_weight.txt")
xs: np.ndarray = np.loadtxt(path)
print(f"{xs.shape=}")

small_xs: np.ndarray = xs[:500]

# fig. 3-16
plt.xlabel("Hiehgt (cm)")
plt.ylabel("Weight (kg)")
plt.scatter(small_xs[:, 0], small_xs[:, 1])
plt.show()