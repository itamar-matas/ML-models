import numpy as np
from time import perf_counter
from numpy.typing import ArrayLike

from abc import ABC, abstractmethod
from typing import List, Dict

from metrics import *

# models
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        pass

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:
        pass

    def score(self, X_test: ArrayLike, Y_test: ArrayLike, scores: List[str] | None = None, onlyvalues=False) -> Dict | List:
        start_time  = perf_counter()
        Y_pred      = self.predict(X_test)
        end_time    = perf_counter()
        
        if scores is None:
            scores = ["accuracy", "runtime"]

        evaluation = {
            "runtime" : lambda a, b: round(end_time - start_time, 4),
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix,
            "recall": recall,
            "precision": precision,
            "f1": F1_score
        }

        report = {s: evaluation[s](Y_test, Y_pred) for s in scores if s in evaluation}

        if onlyvalues:
            return list(report.values())
        
        return report
    
class BaseClassifier(BaseModel):
    @abstractmethod
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        pass
