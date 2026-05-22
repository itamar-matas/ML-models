import numpy as np
import pandas as pd

from typing import List
from abc import ABC, abstractmethod
import sys

_eps = sys.float_info.epsilon

class Scaler(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray | pd.DataFrame, columns: List[str] | List[int] | None = None) -> None: ...

    def transform(self, data: np.ndarray | pd.DataFrame, columns: List[str] | List[int] | None = None) -> np.ndarray | pd.DataFrame:
        if isinstance(data, np.ndarray) and columns is not None and not all([isinstance(col_num, int) for col_num in columns]):
            raise ValueError("When given a numpy.ndarray columns must only contain intager indices")   

        subset = data[columns].copy() if columns is not None else data.copy()
        res = data.copy()

        normalized = self._trans_method(subset)
    
        if columns is not None: res[columns] = normalized
        else: res = normalized
    
        return res

    def fit_transform(self, data: np.ndarray | pd.DataFrame, columns: List[str] | List[int] | None = None) -> np.ndarray | pd.DataFrame:
        self.fit(data, columns)
        return self.transform(data, columns)
    
    @abstractmethod
    def _trans_method(self, subset: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame: ...
    
class minmax(Scaler):
    def __init__(self):
        super().__init__()
        self._min = None
        self._max = None

    def fit(self, data, columns = None) -> None:
        self._min = data.min(axis=0) if columns is None else data[columns].min(axis=0)
        self._max = data.max(axis=0) if columns is None else data[columns].max(axis=0)

    def _trans_method(self, subset):
        return (subset - self._min) / (self._max - self._min + _eps)

class z_score(Scaler):
    def __init__(self):
        super().__init__()
        self._mean = None
        self._std  = None

    def fit(self, data, columns = None) -> None:
        self._mean = data.mean(axis=0) if columns is None else data[columns].mean(axis=0)
        self._std  = data.std(axis=0, ddof=0) if columns is None else data[columns].std(axis=0, ddof=0)

    def _trans_method(self, subset):
        return (subset - self._mean) / (self._std + _eps)

    