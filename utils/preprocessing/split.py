import numpy as np
import pandas as pd

def train_test_split(X: np.ndarray | pd.DataFrame, Y: np.ndarray | pd.DataFrame, ratio: tuple[float, float] = (70, 30), shuffle: bool = True, random_state: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sum(ratio) != 100:
        raise ValueError("train-test ratio must sum up to 100%")
    
    n_samples = X.shape[0]
    indeces = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(indeces)

    split_point = int(ratio[0]/100 * n_samples)

    train_indices = indeces[:split_point]
    test_indices  = indeces[split_point:]

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.iloc[train_indices], Y.iloc[train_indices], X.iloc[test_indices], Y.iloc[test_indices]
    
    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]
