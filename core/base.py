import numpy as np
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Protocol, runtime_checkable

from metrics import *

# ======================================================================================================
#                                               FRAMEWORK                                               
# ======================================================================================================

class BaseEstimator(ABC):
    pass
        
# ======================================================================================================
#                                             PREPROCESSING 
# ======================================================================================================

@runtime_checkable
class TransformerProtocol(Protocol):
    def fit(self, X: ArrayLike, Y: ArrayLike | None = None, **kwargs) -> None: ...

    def transform(self, X: ArrayLike) -> np.ndarray: ...
        
@runtime_checkable
class InversableTransformerProtocol(TransformerProtocol, Protocol):
    def inverse_transform(self, X: ArrayLike) -> np.ndarray: ...

class FitTransformMixin:
    def fit_transform(self, X: ArrayLike, Y: ArrayLike | None = None, **kwargs):
        self.fit(X, Y, **kwargs)
        return self.transform(X)

# ======================================================================================================
#                                                MODELS                                                  
# ======================================================================================================

@runtime_checkable
class ModelProtocol(Protocol):
    def fit(self, X: ArrayLike, Y: ArrayLike | None = None, **kwargs) -> None: ...

    def predict(self, X: ArrayLike) -> np.ndarray: ...

    def score(self, X: ArrayLike, Y: ArrayLike, scores: List[str], onlyvalues: bool = False) -> np.ndarray: ...

@runtime_checkable
class ClassifierProtocol(ModelProtocol, Protocol):
    def predict_proba(self, X: ArrayLike) -> np.ndarray: ...

class ClassifierScoreMixin:
    def score(self, X: ArrayLike, Y: ArrayLike, scores: List[str] | None = None, onlyvalues: bool = False) -> Dict | List:
        Y_pred = self.predict(X)
        
        if scores is None:
            scores = ["accuracy"]

        evaluation = {
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "recall": recall,
            "precision": precision,
            "f1": F1_score
        }

        report = {s: evaluation[s](Y, Y_pred) for s in scores if s in evaluation}

        if onlyvalues:
            return list(report.values())
        
        return report


# ======================================================================================================
#                                               OPTIMIZERS                                                  
# ======================================================================================================

class BaseOptimizer(ABC):
    @abstractmethod
    def __init__(self, theta0: np.ndarray, learning_rate: int = 0.01, **kwargs) -> None: ...

    @abstractmethod
    def step(self, gradient: Callable[[np.ndarray], np.ndarray], **kwargs) -> np.ndarray: ...

    @abstractmethod
    def run(self, loss: Callable[[np.ndarray], float], gradient: Callable[[np.ndarray], np.ndarray]) -> np.ndarray: ...
