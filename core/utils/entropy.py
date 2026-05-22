import numpy as np
import sys

from .probability import contingency_table
_eps = sys.float_info.epsilon

def entropy(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_features = X.shape[1]
    n_samples = X.shape[0]
    entropies = np.zeros(n_features)

    for i in range(n_features):
        _, counts = np.unique(X[:, i], return_counts=True)
        probs = counts / n_samples
        entropies[i] = -np.sum(probs * np.log2(probs + _eps))
        
    return entropies

def conditional_entropy(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    y_labels, y_indexed = np.unique(Y, return_inverse=True)
    n_classes = len(y_labels)
    
    cond_entropies = np.zeros(n_features)
    
    for i in range(n_features):
        feature_col = X[:, i]

        f_labels, f_indexed = np.unique(feature_col, return_inverse=True)
        n_categories = len(f_labels)

        contingency_matrix = contingency_table(f_indexed, y_indexed, num_classes=n_classes, num_categories=n_categories)
        row_sums = contingency_matrix.sum(axis=1)

        probs = contingency_matrix / (row_sums[:, np.newaxis] + _eps)

        category_entropies = -np.sum(probs * np.log2(probs + _eps), axis=1)

        category_weights = row_sums / n_samples
        cond_entropies[i] = np.sum(category_weights * category_entropies)
        
    return cond_entropies

def information_gain(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return entropy(Y) - conditional_entropy(Y, X)