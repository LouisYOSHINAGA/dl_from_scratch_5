import numpy as np
import matplotlib.pyplot as plt

rng: np.random.Generator = np.random.default_rng()

N: int = 1
Ns: list[int] = [1, 2, 4, 10]


def rand_mean(N: int) -> list[np.float64]:
    x_means: list[np.float64] = []
    for _ in range(10000):
        xs: list[float] = [rng.random() for _ in range(N)]
        x_means.append(np.mean(xs))
    return x_means


# fig. 1-9
plt.title(f"N = {N}")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.xlim(-0.05, 1.05)
plt.ylim(0, 5)
plt.hist(rand_mean(N), bins="auto", density=True)
plt.show()

# fig. 1-10
rows: int = 2
cols: int = 2
fig, ax = plt.subplots(rows, cols, figsize=(10, 5))
assert isinstance(ax, np.ndarray)
for i, N in enumerate(Ns):
    ax[i//cols, i%cols].set_title(f"N = {N}")
    ax[i//cols, i%cols].set_xlabel("x")
    ax[i//cols, i%cols].set_ylabel("Probability Density")
    ax[i//cols, i%cols].set_xlim(-0.05, 1.05)
    ax[i//cols, i%cols].set_ylim(0, 5)
    ax[i//cols, i%cols].hist(rand_mean(N), bins="auto", density=True)
plt.tight_layout()
plt.show()