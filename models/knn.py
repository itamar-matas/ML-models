import numpy as np
import pandas as pd

from core.base import BaseModel
from core.utils import L2

class KNNClassifier(BaseModel):
    def __init__(self, k: int, distance_func=L2, voting_weight=None) -> None:
        super().__init__()
        
        self._k = k

        self._dist = distance_func
        self._voting_weight = voting_weight
        self._feature_weights = None

        self._X_train = None
        self._Y_train = None

    def fit(self, X, Y, feature_weights=None) -> None:
        self._X_train = np.array(X)
        self._Y_train = np.array(Y).flatten()

        if self._X_train.ndim == 1: 
            self._X_train = self._X_train.reshape(1, -1)

        if self._X_train.shape[0] != self._Y_train.size or self._X_train.ndim != 2:
            raise ValueError("Dataset shape mismatch")
        
        if feature_weights is not None:
            self._feature_weights = np.asanyarray(feature_weights)

            if self._feature_weights.ndim != 1: 
                raise ValueError("feature_weights must be a 1D vector")
        
            if self._feature_weights.size != self._X_train.shape[1]: 
                raise ValueError("feature_weights must contain the weight of each feature")
            
            self._X_train = self._X_train * self._feature_weights

    def predict(self, X_query) -> tuple[np.ndarray, np.ndarray]:
        X_query = np.array(X_query)

        if X_query.ndim == 1: 
            X_query = X_query.reshape(1, -1)
        
        if self._feature_weights is not None:
            X_query = X_query * self._feature_weights

        preds, confs = [], []
        for x in X_query:
            distances = self._dist(self._X_train, x)
            knn_idx = distances.argpartition(self._k)[:self._k]

            if self._voting_weight is not None:
                weights = self._voting_weight(distances[knn_idx])
            else:
                weights = None
            
            label, confidence = self._vote(self._Y_train[knn_idx], weights)
            preds.append(label)
            confs.append(confidence)
        
        return np.array(preds), np.array(confs)   

    def _vote(self, labels, weights=None):
        if weights is None:
            classes, counts = np.unique(labels, return_counts=True)
            winner_idx = counts.argmax()

            confidence = counts[winner_idx] / self._k
            return classes[winner_idx], confidence
        
        else:
            unique_classes = np.unique(labels)
            scores = {}
        
            for cls in unique_classes:
                cls_weights = weights[labels == cls]
                scores[cls] = np.sum(cls_weights)
        
            winner = max(scores, key=scores.get)
            confidence = scores[winner] / np.sum(weights)

            return winner, confidence