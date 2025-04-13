import os
import numpy as np
import matplotlib.pyplot as plt

rng: np.random.Generator = np.random.default_rng()

path: str = os.path.join(os.path.dirname(__file__), "height.txt")
xs: np.ndarray = np.loadtxt(path)

mu: np.float64 = np.mean(xs)
sigma: np.float64 = np.std(xs)
samples: np.ndarray = rng.normal(mu, sigma, 10000)

# fig. 2-8
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.hist(xs, bins="auto", density=True, alpha=0.7, label="original")
plt.hist(samples, bins="auto", density=True, alpha=0.7, label="generated")
plt.legend()
plt.show()