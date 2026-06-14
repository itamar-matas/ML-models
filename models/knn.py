import numpy as np

from typing import Callable
from numpy.typing import ArrayLike

from core.base import BaseEstimator, ClassifierScoreMixin
from core.utils import L2

class KNNClassifier(BaseEstimator, ClassifierScoreMixin):
    def __init__(self, k: int = 5, distance_func: Callable = L2, voting_weight: Callable | None = None) -> None:
        super().__init__()
        
        self.k = k
        self.distance_func = distance_func
        self.voting_weight = voting_weight

        self._feature_weights = None
        self._classes = None

        self._X_train = None
        self._Y_train = None

    def fit(self, X: ArrayLike, Y: ArrayLike, feature_weights: ArrayLike | None = None) -> None:
        self._X_train = np.atleast_2d(np.asarray(X))
        self._Y_train = np.asarray(Y).ravel()

        if self._X_train.shape[0] != self._Y_train.size:
            raise ValueError("Dataset shape mismatch")
        
        if feature_weights is not None:
            self._feature_weights = np.asarray(feature_weights)

            if self._feature_weights.ndim != 1: 
                raise ValueError("feature_weights must be a 1D array")
        
            if self._feature_weights.size != self._X_train.shape[1]: 
                raise ValueError("feature_weights must contain the weight of each feature")
            
            self._X_train = self._X_train * self._feature_weights

        self._classes = np.unique(self._Y_train)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)   

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        x = np.atleast_2d(np.asarray(X))
        
        if self._feature_weights is not None:
            x = x * self._feature_weights

        probs = np.empty(shape=(x.shape[0], self._classes.size))
        for i, sample in enumerate(x):
            distances = self.distance_func(self._X_train, sample)
            knn_idx = distances.argpartition(self.k)[:self.k]

            if self.voting_weight is not None:
                weights = self.voting_weight(distances[knn_idx])
            else:
                weights = None
            
            prob = self._vote(self._Y_train[knn_idx], weights)
            probs[i] = prob
        
        return probs

    def _vote(self, labels, weights=None):    
        nearest_neighbors, counts = np.unique(labels, return_counts=True)
        probs = np.zeros(self._classes.size)
        
        if weights is None:
            probs[nearest_neighbors] = counts
            return probs / self.k
        
        else:
            for n in nearest_neighbors:
                probs[n] = np.sum(weights[labels == n])

            return probs / np.sum(weights)
    