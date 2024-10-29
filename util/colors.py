# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import numpy as np


BLUE = np.array([1.0, 0.0, 0.0], dtype=np.float64)
GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float64)
RED = np.array([0.0, 0.0, 1.0], dtype=np.float64)
GRAY = np.array([0.5, 0.5, 0.5], dtype=np.float64)
WHITE = np.ones(3, dtype=np.float64)
BLACK = np.zeros(3, dtype=np.float64)


def u8(c: np.ndarray) -> list:
    return np.rint(c * 255).astype(np.uint8).tolist()

def mix(*colors: np.ndarray, weights=None) -> list:
    return np.average(colors, weights=weights, axis=0)
