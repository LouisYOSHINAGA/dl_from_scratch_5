import os
import numpy as np
import matplotlib.pyplot as plt


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

def likelihood(xs: np.ndarray, phis: np.ndarray, mus: np.ndarray, covs: np.ndarray, eps: float =1e-8) -> float:
    # \frac{1}{N} \sum_{n=1}^{N} \log{ \sum_{k=1}^{K} \phi_k \mathcal{N}(x \mid \mu_k, \Sigma_k) }
    L: float = 0
    for x in xs:
        y: float = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / len(xs)


MAX_ITERS: int = 100
THESHOLD: float = 1e-4

path: str = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs: np.ndarray = np.loadtxt(path)
print(f"{xs.shape=}")

phis: np.ndarray = np.array([0.5, 0.5])
mus: np.ndarray = np.array([[0.0, 50.0], [0.0, 100.0]])
covs: np.ndarray = np.array([np.eye(2), np.eye(2)])

K: int = len(phis)
N: int = len(xs)

current_likelihood: float = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E-step
    qs: np.ndarray = np.zeros((N, K))  # q^{(n)}(k)
    for n in range(N):
        x: np.ndarray = xs[n]
        for k in range(K):
            phi: float = phis[k]
            mu: np.ndarray = mus[k]
            cov: np.ndarray = covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)

    # M-step
    qs_sum: np.ndarray = qs.sum(axis=0)  # \sum_{n=1}^{N} q^{(n)}(\cdot)
    for k in range(K):
        # \phi_{k} = \frac{1}{N} \sum_{n=1}^{N} q^{(n)}(k)
        phis[k] = qs_sum[k] / N

        # \mu_{k} = ( \sum_{n=1}^{N} q^{(n)}(k) x^{(n)} ) / ( \sum_{n=1}^{N} q^{(n)}(k) )
        c: float = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        # \Sigma_{k} = ( \sum_{n=1}^{N} q^{(n)}(k) (x^{(n)} - \mu_{k}) (x^{(n)} - \mu_{k})^{T})
        #            / ( \sum_{n=1}^{N} q^{(n)}(k) )
        c = 0
        for n in range(N):
            z: np.ndarray = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]

    print(f"{current_likelihood=:.3f}")

    next_likelihood: float = likelihood(xs, phis, mus, covs)
    if abs(next_likelihood - current_likelihood) < THESHOLD:
        break
    else:
        current_likelihood = next_likelihood

print(f"{phis=}")
print(f"{mus=}")
print(f"{covs=}")


def plot_contour(w: np.ndarray, mus: np.ndarray, covs: np.ndarray) -> None:
    x: np.ndarray = np.arange(1, 6, 0.1)
    y: np.ndarray = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z: np.ndarray = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x: np.ndarray = np.array([X[i, j], Y[i, j]])
            for k in range(K):
                mu: np.ndarray = mus[k]
                cov: np.ndarray = covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)

# fig. 5-11
plt.xlabel("Eruptions (Min)")
plt.ylabel("Waiting (Min)")
plt.scatter(xs[:, 0], xs[:, 1])
plot_contour(phis, mus, covs)
plt.tight_layout()
plt.show()