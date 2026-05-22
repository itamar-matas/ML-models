import numpy as np

def L1(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = np.atleast_2d(mat)
    vec = np.atleast_2d(vec)

    return np.abs(mat - vec).sum(axis=1)

def L2(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = np.atleast_2d(mat)
    vec = np.atleast_2d(vec)

    return np.sqrt(((mat - vec) ** 2).sum(axis=1))

def L_inf(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    mat = np.atleast_2d(mat)
    vec = np.atleast_2d(vec)

    return np.abs(mat - vec).max(axis=1)