import numpy as np
import matplotlib.pyplot as plt

rng: np.random.Generator = np.random.default_rng()

N: int = 5


def normal(x: np.ndarray, mu: float =0.0, sigma: float =1.0) -> np.ndarray:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))


if __name__ == "__main__":
    x_sums: list[np.float64] = []
    for _ in range(10000):
        xs: list[float] = [rng.random() for _ in range(N)]
        x_sums.append(np.sum(xs))

    x_norm: np.ndarray = np.linspace(-5, 5, 1000)
    y_norm: np.ndarray = normal(x_norm, mu=N/2, sigma=np.sqrt(N/12))

    # fig. 1-12
    plt.title(f"N = {N}")
    plt.xlim(-1, 6)
    plt.hist(x_sums, bins="auto", density=True)
    plt.plot(x_norm, y_norm)
    plt.show()