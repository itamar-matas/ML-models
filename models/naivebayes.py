import numpy as np
import pandas as pd

from core.base import BaseModel
from core.utils import discrete_priors, discrete_likelihood

# for now categorical data
class NaiveBayesClassifier(BaseModel):
    def __init__(self, alpha: int = 1) -> None:
        super().__init__()
        
        self._alpha           = alpha
        self._logpriors       = None
        self._loglikelihoods  = None

    def fit(self, X, Y) -> None:
        x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()

        self._logpriors = np.log(discrete_priors(y))
        self._loglikelihoods = [
            np.log(prob_table) for prob_table in discrete_likelihood(x, y, alpha=self._alpha)
        ]

    def predict(self, X):
        X = np.atleast_2d(np.array(X))
        
        n_samples, n_features = X.shape
        n_classes = len(self._logpriors)

        logposteriors = np.zeros(shape=(n_samples, n_classes)) + self._logpriors
        for i in range(n_features):
            feature_col = X[:, i]
            logposteriors += self._loglikelihoods[i][feature_col]

        return np.argmax(logposteriors, axis=1)

