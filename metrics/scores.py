import numpy as np
import sys
    
_eps = sys.float_info.epsilon

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, ndigits: int = 4) -> float: 
    return np.round(np.mean(y_pred == y_true), ndigits)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    classes = np.unique(np.concatenate([y_true, y_pred]))
    N = len(classes)

    y_t_idx = np.searchsorted(classes, y_true)
    y_p_idx = np.searchsorted(classes, y_pred)

    conf_mat = np.zeros(shape=(N, N), dtype=int)
    np.add.at(conf_mat, (y_t_idx, y_p_idx), 1)

    return conf_mat

def recall(y_true: np.ndarray, y_pred: np.ndarray, ndigits: int = 4) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    return np.round(cm.diagonal() / (cm.sum(axis=1) + _eps), ndigits)


def precision(y_true: np.ndarray, y_pred: np.ndarray, ndigits: int = 4) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    return np.round(cm.diagonal() / (cm.sum(axis=0) + _eps), ndigits)

def F1_score(y_true: np.ndarray, y_pred: np.ndarray, ndigits: int = 4) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)

    recl = np.round(cm.diagonal() / (cm.sum(axis=1) + _eps), ndigits)
    prec = np.round(cm.diagonal() / (cm.sum(axis=0) + _eps), ndigits)

    return round(2 * prec * recl / (prec + recl + _eps), ndigits)