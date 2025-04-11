import os
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeAlias

Float: TypeAlias = float | np.float64


def normal(x: np.ndarray, mu: Float =0.0, sigma: Float =1.0) -> np.ndarray:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))


path: str = os.path.join(os.path.dirname(__file__), "height.txt")
xs: np.ndarray = np.loadtxt(path)

x: np.ndarray = np.linspace(150, 190, 1000)
y: np.ndarray = normal(x, mu=np.mean(xs), sigma=np.std(xs))

# fig. 2-6
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.hist(xs, bins="auto", density=True)
plt.plot(x, y)
plt.show()