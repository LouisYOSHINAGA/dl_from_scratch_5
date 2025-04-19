import numpy as np
import matplotlib.pyplot as plt

rng: np.random.Generator = np.random.default_rng()

# trained parameter
mus: np.ndarray = np.array([
    [ 2.00, 54.50],
    [ 4.30, 80.00]
])
covs: np.ndarray = np.array([
    [[ 0.07,  0.44],
     [ 0.44, 33.70]],
    [[ 0.17,  0.94],
     [ 0.94, 36.00]]
])
phis: np.ndarray = np.array([0.35, 0.65])

def sample() -> np.ndarray:
    z: int = rng.choice(2, p=phis)
    mu: np.ndarray = mus[z]
    cov: np.ndarray = covs[z]
    x: np.ndarray = rng.multivariate_normal(mu, cov)
    return x


N: int = 500
xs: np.ndarray = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample()

# fig. 4-6
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(xs[:, 0], xs[:, 1], color="orange", alpha=0.7)
plt.show()