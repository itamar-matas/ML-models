import numpy as np

from numpy.typing import ArrayLike

from core.base import BaseEstimator, ClassifierScoreMixin
from core.utils import discrete_priors, discrete_likelihood

# for now categorical data
class NaiveBayesClassifier(BaseEstimator, ClassifierScoreMixin):
    def __init__(self, alpha: int = 1) -> None:
        super().__init__()
        
        self.alpha = alpha

        self._classes         = None
        self._logpriors       = None
        self._loglikelihoods  = None

    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        x, y = np.atleast_2d(np.asarray(X)), np.asarray(Y).ravel()

        self._logpriors = np.log(discrete_priors(y))
        self._loglikelihoods = [
            np.log(prob_table) for prob_table in discrete_likelihood(x, y, alpha=self.alpha)
        ]

    def predict(self, X: ArrayLike):
        x = np.atleast_2d(np.asarray(X))

        logposteriors = np.zeros(shape=(x.shape[0], len(self._logpriors))) + self._logpriors
        for i in range(x.shape[1]):
            feature_col = X[:, i]
            logposteriors += self._loglikelihoods[i][feature_col]

        return np.argmax(logposteriors, axis=1)

    def predict_proba(self, X: ArrayLike):
        x = np.atleast_2d(np.asarray(X))

        logposteriors = np.zeros(shape=(x.shape[0], len(self._logpriors))) + self._logpriors
        for i in range(x.shape[1]):
            feature_col = x[:, i]
            logposteriors += self._loglikelihoods[i][feature_col]

        logposteriors -= logposteriors.max(axis=1, keepdims=True)
        posteriors = np.exp(logposteriors)
        return posteriors / posteriors.sum(axis=1, keepdims=True)
