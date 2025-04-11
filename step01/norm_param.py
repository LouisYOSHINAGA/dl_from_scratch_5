import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, mu: float =0.0, sigma: float =1.0) -> np.ndarray:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))


x: np.ndarray = np.linspace(-10, 10, 1000)

y0: np.ndarray = normal(x, mu=-3)
y1: np.ndarray = normal(x, mu=0)
y2: np.ndarray = normal(x, mu=5)

# fig. 1-6
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y0, label="$\\mu$=-3")
plt.plot(x, y1, label="$\\mu$=0")
plt.plot(x, y2, label="$\\mu$=5")
plt.legend()
plt.show()


y0: np.ndarray = normal(x, sigma=0.5)
y1: np.ndarray = normal(x, sigma=1)
y2: np.ndarray = normal(x, sigma=2)

# fig. 1-7
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y0, label="$\\sigma$=0.5")
plt.plot(x, y1, label="$\\sigma$=1")
plt.plot(x, y2, label="$\\sigma$=2")
plt.legend()
plt.show()
