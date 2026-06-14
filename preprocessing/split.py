import numpy as np
from numpy.typing import ArrayLike

def train_test_split(X: ArrayLike, Y: ArrayLike, ratio: tuple[float, float] = (70, 30), shuffle: bool = True, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()
    _check_xy(x, y)
    
    total = sum(ratio)
    splits_points = np.cumsum(ratio) / total * y.size

    indices = _split_indices(y.size, splits_points.astype(int)[:-1], shuffle, RS=random_state)
    
    return (
        x[indices[0]], x[indices[1]], 
        y[indices[0]], y[indices[1]]
    )

def train_valid_test_split(X: ArrayLike, Y: ArrayLike, ratio: tuple[float, float] = (50, 30, 20), shuffle: bool = True, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()
    _check_xy(x, y)

    total = sum(ratio)
    splits_points = np.cumsum(ratio) / total * y.size

    indices = _split_indices(y.size, splits_points.astype(int)[:-1], shuffle, RS=random_state)
    
    return (
        x[indices[0]], x[indices[1]], x[indices[2]],
        y[indices[0]], y[indices[1]], y[indices[2]]
    )

class kFolds:
    def __init__(self, k: int = 5, shuffle: bool = True, random_state: int | None = None):
        self._k = k
        self._shuffle = shuffle
        self._RS = random_state
        
    def split(self, X: ArrayLike, Y : ArrayLike | None = None, groups=None):
        x = np.atleast_2d(np.asarray(X))
        n_samples = x.shape[0]

        indices = _split_indices(n_samples, 1, self._shuffle, self._RS)[0]

        for i in range(self._k):
            start, end = n_samples * i // self._k, n_samples * (i+1) // self._k

            valid_mask = np.zeros(n_samples, dtype=bool)
            valid_mask[start:end] = True

            yield indices[~valid_mask], indices[valid_mask]

        
    def get_k_splits(self) -> int: return self._k


def _split_indices(length: int, splits: np.ndarray | int, shuffle: bool, RS: int | None):
    indices = np.arange(length)

    if shuffle:
        rng = np.random.default_rng(seed=RS)
        rng.shuffle(indices)

    return np.split(indices, splits)

def _check_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim > 2:
        raise ValueError("X must be 2D")
    
    if x.shape[0] != y.size:
        raise ValueError("X and y must have the same number of samples")
    