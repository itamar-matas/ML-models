import numpy as np

def train_test_split(X, Y, ratio=(70, 30)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if ratio[0] + ratio[1] != 100:
        raise ValueError("train-test ratio must sum up to 100%")
    
    else:
        indeces = list(range(len(X)))
        np.random.shuffle(indeces)

        train_indeces = indeces[int(ratio[0]/100 * len(X)):]
        test_indeces  = indeces[:int(ratio[0]/100 * len(X))]

        return X[train_indeces], Y[train_indeces], X[test_indeces], Y[test_indeces]
