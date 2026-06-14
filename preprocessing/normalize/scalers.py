import numpy as np

from numpy.typing import ArrayLike
from core.base import BaseEstimator, InversableTransformerProtocol, FitTransformMixin
import sys

_eps = sys.float_info.epsilon
    
class minmax(BaseEstimator, InversableTransformerProtocol, FitTransformMixin):
    def __init__(self):
        super().__init__()

        self._min = None
        self._max = None

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None) -> None:
        arr = np.asarray(X)

        self._min = arr.min(axis=0)
        self._max = arr.max(axis=0)

    def transform(self, X: ArrayLike):
        return (np.asarray(X) - self._min) / (self._max - self._min + _eps)
    
    def inverse_transform(self, X: ArrayLike):
        return np.asarray(X) * (self._max - self._min + _eps) + self._min

class z_score(BaseEstimator, InversableTransformerProtocol, FitTransformMixin):
    def __init__(self):
        super().__init__()

        self._mean = None
        self._std  = None

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None) -> None:
        arr = np.asarray(X)

        self._mean = arr.mean(axis=0)
        self._std  = arr.std(axis=0, ddof=0)

    def transform(self, X: ArrayLike):
        return (np.asarray(X) - self._mean) / (self._std + _eps)
    
    def inverse_transform(self, X: ArrayLike):
        return np.asarray(X) * (self._std + _eps) + self._mean

    