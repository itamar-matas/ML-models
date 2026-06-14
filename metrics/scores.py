import numpy as np
import sys
    
_eps = sys.float_info.epsilon

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float: 
    return np.mean(y_pred == y_true)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray: 
    classes = np.unique(np.concatenate([y_true, y_pred]))
    N = len(classes)

    y_t_idx = np.searchsorted(classes, y_true)
    y_p_idx = np.searchsorted(classes, y_pred)

    conf_mat = np.zeros(shape=(N, N), dtype=int)
    np.add.at(conf_mat, (y_t_idx, y_p_idx), 1)

    return conf_mat

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(cm.diagonal() / (cm.sum(axis=1) + _eps))

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    return np.mean(cm.diagonal() / (cm.sum(axis=0) + _eps))

def F1_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)

    recl = cm.diagonal() / (cm.sum(axis=1) + _eps),
    prec = cm.diagonal() / (cm.sum(axis=0) + _eps)

    return np.mean(2 * prec * recl / (prec + recl + _eps))

def mean_confidence(probability_table: np.ndarray) -> np.ndarray:
    ret = np.zeros(probability_table.shape[1])

    predictions = probability_table.argmax(axis=1)
    confidences = probability_table[predictions]
    
    for cls in range(probability_table.shape[1]):
        cls_mask = (predictions == cls)
        
        ret[cls] = confidences[cls_mask].mean() if np.any(cls_mask) else 0.0
            
    return ret