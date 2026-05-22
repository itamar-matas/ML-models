import numpy as np
from numpy.typing import ArrayLike

class LabelEncoder:
    def __init__(self):
        self._classes = None

    def fit(self, Y: ArrayLike) -> None:
        y = np.array(Y).ravel().astype(str)
        self._classes = np.unique(y)

    def transform(self, Y: ArrayLike) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("cannot transform without fitting first")
        
        y = np.array(Y).ravel()
        return self._classes.searchsorted(y)

    def fit_transform(self, Y: ArrayLike) -> np.ndarray:
        y = np.array(Y).ravel()
        self._classes, inverse = np.unique(y, return_inverse=True)
        return inverse

    def inverse_transform(self, Y: ArrayLike) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("cannot transform without fitting first")
        
        y = np.array(Y).ravel()
        return self._classes[y]
                    
class OrdinalEncoder:
    def __init__(self, unknown_value: str = "__unknown__"):
        self._encoders = None
        self._unknown_value = unknown_value

    def fit(self, X: ArrayLike) -> None:
        x = np.atleast_2d(np.asarray(X))
        self._encoders = []

        for i in range(x.shape[1]):
            feature_col = x[:, i]
            encoder = LabelEncoder()

            encoder.fit(feature_col)
            self._encoders.append(encoder)

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self._encoders is None:
            raise RuntimeError("cannot transform without fitting first")

        x = np.atleast_2d(np.asarray(X))
        encoded = np.zeros_like(x, dtype=int)

        for i in range(x.shape[1]):
            le = self._encoders[i]
            feature_col = x[:, i]
            
            coded = le._classes.searchsorted(feature_col)
            safe_coded = np.minimum(coded, len(le._classes) - 1)
            
            mask = (coded < len(le._classes)) & (le._classes[safe_coded] == feature_col)
            encoded[:, i] = np.where(mask, coded, len(le._classes))

        return encoded
    
    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        x = np.atleast_2d(np.asarray(X)) 

        self.fit(x)
        return self.transform(x)     

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        if self._encoders is None:
            raise RuntimeError("cannot transform without fitting first")
        
        x = np.atleast_2d(np.asarray(X))
        inverse = np.full_like(x, self._unknown_value, dtype=object)

        for i in range(x.shape[1]):
            le = self._encoders[i]

            feature_col = x[:, i] 
            safe_col = np.where(feature_col >= len(le._classes), 0, feature_col)

            res = le._classes.astype(object)[safe_col]
            res[feature_col >= len(le._classes)] = self._unknown_value

            inverse[:, i] = res

        return inverse
