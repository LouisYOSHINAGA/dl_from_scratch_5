import numpy as np
import matplotlib.pyplot as plt

def normal(x: float|np.ndarray, mu: float =0.0, sigma: float =1.0) -> float|np.ndarray:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

if __name__ == "__main__":
    x: np.ndarray = np.linspace(-5, 5, 100)
    y: np.ndarray = normal(x)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.show()