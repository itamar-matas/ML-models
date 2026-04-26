import numpy as np
import pandas as pd

from typing import List, Callable
import sys

_eps = sys.float_info.epsilon

def _norm(data: np.ndarray | pd.DataFrame, method: Callable,  columns: List[str] | List[int] | None = None, **kwargs) -> np.ndarray | pd.DataFrame:
    if isinstance(data, np.ndarray) and isinstance(columns, list):
        raise ValueError("When given a numpy.ndarray columns cannot be textual")   

    subset = data[columns].copy() if columns is not None else data.copy()
    norm = data.copy()

    res = method(subset, **kwargs)
    
    if columns is not None: norm[columns] = res
    else: norm = res
    
    return norm

def minmax(data: np.ndarray | pd.DataFrame, columns: List[str] | List[int] | None = None, fixed_min: float | None = None, fixed_max: float | None = None) -> np.ndarray | pd.DataFrame:
    if (fixed_max is not None and fixed_min is None) or (fixed_max is None and fixed_min is not None):
        raise ValueError("Must provide exactly none or both fixed minimum and maximum")    
    
    minmax_method = lambda sbst, mx, mn: (sbst - mn) / (mx - mn + _eps) if mx is not None else (sbst - sbst.min(axis=0)) / (sbst.max(axis=0) - sbst.min(axis=0) + _eps)
    return _norm(data, minmax_method, columns, mx=fixed_max, mn=fixed_min)
    

def z_score(data: np.ndarray | pd.DataFrame, columns: List[str] | List[int] | None = None, fixed_mean: float | None = None, fixed_std: float | None = None) -> np.ndarray | pd.DataFrame:
    if (fixed_mean is not None and fixed_std is None) or (fixed_mean is None and fixed_std is not None):
        raise ValueError("Must provide exactly none or both fixed mean and standard deviation")       

    zscore_method = lambda sbst, mean, std: (sbst - mean) / std if std is not None else (sbst - sbst.mean(axis=0)) / ((sbst.std(ddof=0) + _eps) if isinstance(sbst, pd.DataFrame) else (np.std(sbst, axis=0, ddof=0) + _eps))
    return _norm(data, zscore_method, columns, mean=fixed_mean, std=fixed_std)