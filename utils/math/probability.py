import numpy as np

from typing import List
from collections.abc import Sequence

def discrete_priors(Y: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """
        calculates the probability of each class in indexed array `Y`, i.e., P(class)

        Parameters:
            Y (np.ndarray): a 1D indexed array of class labels
            num_classes (int, Optional): the total number of classes

        Returns:
            priors (np.ndarray): A 1D array of length `num_classes` containing the prior probabilities of each class
    """
    return np.bincount(Y, minlength=num_classes) / len(Y)

def discrete_likelihood(X: np.ndarray, Y: np.ndarray, num_classes: int | None = None, num_categories: Sequence[int] | None = None, alpha: int = 1) -> List[np.ndarray]:
    """
        calculate the probability of each category of each feature in indexed array `X` given a class in `Y`, i.e., P(feature_category | class)

        Parameters:
            X (np.ndarray): a 2D indexed array of shape (n_samples, n_features) of samples
            Y (np.ndarray): a 1D indexed array of shape (n_samples,) of classes
            num_classes (int, Optional): the total number of unique classes in `Y`
            num_categories (Sequence[int], Optional): a 1D sequence containing the number of categories for each feature in `X`
            alpha (int, Optional): the smoothing parameter for Laplace smoothing (default is 1)

        Returns:
            likelihoods (List[np.ndarray]): a list of 2D arrays, where each array corresponds to a feature and has shape (num_categories[i], num_classes) containing the likelihood probabilities for each category of that feature given each class
    """

    if X.ndim == 1:
        X = np.atleast_2d(X)

    n_features = X.shape[1]
    n_classes = num_classes or len(np.unique(Y))

    probabilities = []
    for i in range(n_features):
        feature_col = X[:, i]
        n_cat = num_categories[i] if num_categories is not None else len(np.unique(feature_col))

        contingency_matrix = contingency_table(feature_col.flatten(), Y, num_classes=n_classes, num_categories=n_cat)
        col_sums = contingency_matrix.sum(axis=0)

        probabilities.append((contingency_matrix + alpha) / (col_sums + alpha * n_cat))

    return probabilities

def contingency_table(feature: np.ndarray, Y: np.ndarray, num_classes: int | None = None, num_categories: int | None = None) -> np.ndarray:
    """
        constructs a contingency table for a given feature and class labels

        Parameters:
            feature (np.ndarray): a 1D indexed array of shape (n_samples,) of categories
            Y (np.ndarray): a 1D indexed array of shape (n_samples,) of classes
            num_classes (int, Optional): the total number of unique classes in `Y` 
            num_categories (int, Optional): the total number of unique categories in `feature`

        Returns:
            contingency_table (np.ndarray): a 2D array of shape (num_categories, num_classes) where each entry [i, j] contains the count of samples that belong to category i of the feature and class j
    """

    num_categories = num_categories or len(np.unique(feature))
    num_classes = num_classes or len(np.unique(Y))

    combined_index = feature * num_classes + Y 
    counts = np.bincount(combined_index, minlength=num_categories * num_classes)

    return counts.reshape(num_categories, num_classes)