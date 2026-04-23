import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    classes = sorted(np.unique(y_true))
    N = len(classes)

    y_t_idx = np.searchsorted(classes, y_true)
    y_p_idx = np.searchsorted(classes, y_pred)

    conf_mat = np.zeros(shape=(N, N), dtype=int)
    np.add.at(conf_mat, (y_t_idx, y_p_idx), 1)

    return conf_mat