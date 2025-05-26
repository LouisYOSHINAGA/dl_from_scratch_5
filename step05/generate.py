import os
import numpy as np
import matplotlib.pyplot as plt


rng: np.random.Generator = np.random.default_rng()


def multivariate_normal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    # \mathcal{N}(x \mid \mu, \Sigma)
    det: float = np.linalg.det(cov)
    inv: np.ndarray = np.linalg.inv(cov)
    z: np.ndarray = 1 / np.sqrt((2 * np.pi) ** len(x) * det)
    y: float = z * np.exp(- (x - mu).T @ inv @ (x - mu) / 2.0)  # type: ignore
    return y

def gmm(x: np.ndarray, phis: np.ndarray, mus: np.ndarray, covs: np.ndarray) -> float:
    # \sum_{k=1}^{K} \phi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
    y: float = 0
    for k in range(len(phis)):
        phi: float = phis[k]
        mu: np.ndarray = mus[k]
        cov: np.ndarray = covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


path: str = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
orig_xs: np.ndarray = np.loadtxt(path)

learned_phis: np.ndarray = np.array([0.35595101, 0.64404899])
learned_mus: np.ndarray = np.array([[ 2.03657905, 54.48044568], [ 4.28983008, 79.97014211]])
learned_covs: np.ndarray = np.array([[[ 0.06931941,  0.43676263], [ 0.43676263, 33.70834056]],
                                     [[ 0.16975533,  0.93790633], [ 0.93790633, 36.01589174]]])

N: int = 500

new_xs: np.ndarray = np.zeros((N, 2))
for n in range(N):
    k: int = rng.choice(2, p=learned_phis)
    new_xs[n] = rng.multivariate_normal(learned_mus[k], learned_covs[k])


# fig. 5-12
plt.xlabel("Eruptions (Min)")
plt.ylabel("Waiting (Min)")
plt.scatter(orig_xs[:, 0], orig_xs[:, 1], alpha=0.7, label="Original")
plt.scatter(new_xs[:, 0], new_xs[:, 1], alpha=0.7, label="Generated")
plt.legend()
plt.tight_layout()
plt.show()