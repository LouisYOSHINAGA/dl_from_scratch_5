import os
import numpy as np
import matplotlib.pyplot as plt

path: str = os.path.join(os.path.dirname(__file__), "height.txt")
xs: np.ndarray = np.loadtxt(path)
print(xs.shape)

# fig. 2-4
plt.xlabel("Heihgt (cm)")
plt.ylabel("Probability Density")
plt.hist(xs, bins="auto", density=True)
plt.show()
