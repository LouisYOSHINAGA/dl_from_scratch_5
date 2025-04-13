import os
import numpy as np
from scipy.stats import norm

path: str = os.path.join(os.path.dirname(__file__), "height.txt")
xs: np.ndarray = np.loadtxt(path)
mu: np.float64 = np.mean(xs)
sigma: np.float64 = np.std(xs)

x: float = 160
p1: np.float64 = norm.cdf(x, loc=mu, scale=sigma)  # type: ignore
print(f"p(x <= {x}) = {p1}")

x = 180
p2: np.ndarray = norm.cdf(x, loc=mu, scale=sigma)
print(f"p(x > {x}) = {1-p2}")