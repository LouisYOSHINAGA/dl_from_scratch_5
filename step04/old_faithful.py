import os
import numpy as np
import matplotlib.pyplot as plt

path: str = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs: np.ndarray = np.loadtxt(path)
print(f"{xs.shape=}")
print(f"{xs[0]=}")

# fig. 4-4
plt.xlabel("Eruptions (Min)")
plt.ylabel("Waiting (Min)")
plt.scatter(xs[:, 0], xs[:, 1])
plt.show()