import numpy as np

from numpy.typing import ArrayLike
from typing import Dict, List, Tuple, Callable

from .base import BaseEstimator, ModelProtocol, TransformerProtocol, InversableTransformerProtocol, ClassifierProtocol

class Pipeline(BaseEstimator):
    def __init__(self, steps: List[Tuple[str, BaseEstimator] | BaseEstimator]):
        if len(steps) == 0:
            raise ValueError("Cannot create an empty pipeline")
        
        inversable = True
        named_steps = dict()

        for i, step in enumerate(steps[:-1]):
            if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str) and isinstance(step[1], TransformerProtocol):
                named_steps[step[0]] = step[1]
                steps[i] = step[1]

            elif isinstance(step, TransformerProtocol):
                named_steps[type(step).__name__.lower()] = step

            else:
                raise TypeError(f"Step {i} ({type(step).__name__}) must be a Transformer (inherit from BaseTransformer).")
            
            if not isinstance(steps[i], InversableTransformerProtocol):
                inversable = False

        if isinstance(steps[-1], tuple) and len(steps[-1]) == 2 and isinstance(steps[-1][0], str) and isinstance(steps[-1][1], ModelProtocol):
            named_steps[steps[-1][0]] = steps[-1][1]
            steps[-1] = steps[-1][1]
            
        elif isinstance(steps[-1], ModelProtocol):
            named_steps[type(steps[-1]).__name__.lower()] = steps[-1]

        else:
            raise TypeError(f"The last step ({type(steps[-1]).__name__}) must be a Model (inherit from BaseModel).")
        
        self._steps = steps
        self._named_steps = named_steps
        self._inversable = inversable
        self._fitted = False
        self._classifier = isinstance(steps[-1], ClassifierProtocol)

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None):
        self.fit_transform(X, Y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(action="transfrom")

        x, _ = self._apply_transformers(X, None, operation=lambda x, y, trans: trans.transform(x))
        return x
    
    def fit_transform(self, X: ArrayLike, Y: ArrayLike | None = None) -> np.ndarray:
        x, y = self._apply_transformers(X, Y, operation=lambda x, y, trans: trans.fit_transform(x, y))

        self._steps[-1].fit(x, y)
        self._fitted = True

        return x
    
    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(action="inverse")

        if not self._inversable:
            raise RuntimeError("not all of the transformers are inversable")
        
        x = np.atleast_2d(np.asarray(X))

        for transformer in reversed(self._steps[:-1]):
            x = transformer.inverse_transform(x)

        return x

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(action="predict")

        x, _ = self._apply_transformers(X, None, operation=lambda x, y, trans: trans.transform(x))

        return self._steps[-1].predict(x)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        self._check_fitted(action="predict probabilities")
        
        if not self._classifier:
            raise RuntimeError("cannot predict probabilities with a non-classifier model")

        x, _ = self._apply_transformers(X, None, operation=lambda x, y, trans: trans.transform(x))

        return self._steps[-1].predict_proba(x)
    
    def score(self, X_test: ArrayLike, Y_test: ArrayLike, scores: List[str] | None = None, onlyvalues=False) -> Dict | List:
        self._check_fitted(action="score")

        x, _ = self._apply_transformers(X_test, None, operation=lambda x, y, trans: trans.transform(x))

        return self._steps[-1].score(x, Y_test, scores=scores, onlyvalues=onlyvalues)


    def __getitem__(self, key: str | int) -> TransformerProtocol | ModelProtocol:
        return self._named_steps.get(key) if isinstance(key, str) else self._steps[key]


    def _check_fitted(self, action: str):
        if not self._fitted:
            raise RuntimeError(f"Cannot {action} with an unfitted pipeline")
    
    def _apply_transformers(self, X, Y, operation: Callable):
        x = np.atleast_2d(np.asarray(X))
        y = None if Y is None else np.asarray(Y).ravel()

        for transformer in self._steps[:-1]:
            x = operation(x, y, transformer)

        return x, y