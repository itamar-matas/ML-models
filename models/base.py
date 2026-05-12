from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing import List, Dict

from utils.metrics import accuracy_score, confusion_matrix

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: ArrayLike, Y: ArrayLike):
        pass

    @abstractmethod
    def predict(self, X: ArrayLike):
        pass

    def score(self, X_test: ArrayLike, Y_test: ArrayLike, scores: List[str] | None = None, onlyvalues=False) -> Dict | List:
        Y_pred = self.predict(X_test)
        
        if scores is None:
            scores = ["accuracy"]

        evaluation = {
            "accuracy": accuracy_score,
            "confusion_matrix": confusion_matrix
        }

        report = {s: evaluation[s](Y_test, Y_pred) for s in scores if s in evaluation}

        if onlyvalues:
            return list(report.values())
        
        return report