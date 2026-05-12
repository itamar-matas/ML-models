import numpy as np
import pandas as pd

from .base import BaseModel

from utils.math import discrete_priors, discrete_likelihood

# for now only indexed categorical data
class NaiveBayes(BaseModel):
    def __init__(self, alpha: int = 1) -> None:
        super().__init__()
        
        self.alpha           = alpha
        self.logpriors       = None
        self.loglikelihoods  = None

        self.feature_mapping = []
        self.class_mapping   = None

    def fit(self, X, Y) -> None:
        X, Y = np.atleast_2d(np.array(X)), np.array(Y).flatten()

        self.class_mapping, Y_idx = np.unique(Y, return_inverse=True)

        n_cls = len(self.class_mapping)
        self.logpriors = np.log(discrete_priors(Y_idx, num_classes=n_cls))

        n_cats = []
        X_idx = np.zeros_like(X, dtype=int)

        for i in range(X.shape[1]):
            categories, col_idx = np.unique(X[:, i], return_inverse=True)
            X_idx[:, i] = col_idx

            n_cats.append(len(categories))
            self.feature_mapping.append(categories)

        self.loglikelihoods = [
            np.log(prob_table) for prob_table in discrete_likelihood(X_idx, Y_idx, alpha=self.alpha, num_classes=n_cls, num_categories=n_cats)
        ]

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X = np.atleast_2d(np.array(X))
        
        n_samples, n_features = X.shape
        n_classes = len(self.logpriors)

        logposteriors = np.zeros(shape=(n_samples, n_classes)) + self.logpriors
        for i in range(n_features):
            mapping = self.feature_mapping[i]
            loglikelihood_table = self.loglikelihoods[i]

            feature_idx = np.searchsorted(mapping, X[:, i])
            known = (feature_idx < len(mapping)) & (mapping[np.clip(feature_idx, 0, len(mapping)-1)] == X[:, i])

            logposteriors[known] += loglikelihood_table[feature_idx[known]]

        classes_idx = np.argmax(logposteriors, axis=1)
        return self.class_mapping[classes_idx]

