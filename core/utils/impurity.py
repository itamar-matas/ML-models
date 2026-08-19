import numpy as np
import sys

from typing import Callable, Tuple

from .probability import contingency_table
_eps = sys.float_info.epsilon

entropy_method = lambda P: -np.sum(P * np.log2(P + _eps), axis=1)
gini_method    = lambda P: 1 - np.sum(P ** 2, axis=1)

def entropy(X: np.ndarray) -> np.ndarray:
    return _impurity_measure(X, method=entropy_method)

def gini(X: np.ndarray) -> np.ndarray:
    return _impurity_measure(X, method=gini_method)


def information_gain(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return entropy(Y) - _split_impurity_measure(Y, X, method=entropy_method)

def gini_gain(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return gini(Y) - _split_impurity_measure(Y, X, method=gini_method)

def best_threshold_and_gain(Y: np.ndarray, X: np.ndarray, method: Callable) -> Tuple[np.ndarray, np.ndarray]:
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    n_samples = X.shape[0]

    classes, classes_count = np.unique(Y, return_counts=True)
    impurity = method(classes_count[np.newaxis, :] / n_samples)[0]

    thresholds = np.zeros(n_features, dtype=np.float64)
    gains = np.zeros(n_features)
        
    for i in range(n_features):
        feature_col = X[:, i]
    
        sort_idx = np.argsort(feature_col)
        X_col_sorted = feature_col[sort_idx]
        Y_sorted = Y[sort_idx]

        if X_col_sorted[0] == X_col_sorted[-1]:
            thresholds[i], gains[i] = 0.0, 0.0
            continue

        split_indices = np.where(X_col_sorted[:-1] != X_col_sorted[1:])[0]

        Y_onehot = (Y_sorted[:, np.newaxis] == classes).astype(int)
    
        left_counts = np.cumsum(Y_onehot, axis=0)[split_indices]
        right_counts = classes_count - left_counts

        n_left = split_indices + 1
        n_right = n_samples - n_left

        p_left = left_counts / n_left[:, np.newaxis]
        p_right = right_counts / n_right[:, np.newaxis]

        conditional_impurity = (n_left / n_samples) * method(p_left) + (n_right / n_samples) * method(p_right)

        best_idx = np.argmin(conditional_impurity)

        thresholds[i] = (X_col_sorted[split_indices[best_idx]] + X_col_sorted[split_indices[best_idx] + 1]) / 2.0
        gains[i] = impurity - conditional_impurity[best_idx]

    return thresholds, gains
    

def _impurity_measure(X: np.ndarray, method: Callable) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_features = X.shape[1]
    n_samples = X.shape[0]
    measures = np.zeros(n_features)

    for i in range(n_features):
        _, counts = np.unique(X[:, i], return_counts=True)
        probs = counts / n_samples
        measures[i] = method(probs[np.newaxis, :])[0]
        
    return measures
    
def _split_impurity_measure(Y: np.ndarray, X: np.ndarray, method: Callable) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    y_labels, y_indexed = np.unique(Y, return_inverse=True)
    n_classes = len(y_labels)
    
    split_measures = np.zeros(n_features)
    
    for i in range(n_features):
        feature_col = X[:, i]

        f_labels, f_indexed = np.unique(feature_col, return_inverse=True)
        n_categories = len(f_labels)

        contingency_matrix = contingency_table(f_indexed, y_indexed, num_classes=n_classes, num_categories=n_categories)
        row_sums = contingency_matrix.sum(axis=1)

        probs = contingency_matrix / (row_sums[:, np.newaxis] + _eps)

        category_measures = method(probs)

        category_weights = row_sums / n_samples
        split_measures[i] = np.sum(category_weights * category_measures)
        
    return split_measures

