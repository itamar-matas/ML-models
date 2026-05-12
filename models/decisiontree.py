import numpy as np
import pandas as pd

from typing import Dict, List
from .base import BaseModel

from utils.math import entropy, information_gain


#ID3 - only base desion tree for the moment, only categorical features.
class DecisionTree(BaseModel):
    def __init__(self, max_depth: int | None = None, min_split: int = 2, min_gain: float = 1e-4) -> None:
        super().__init__()

        self._max_depth = max_depth
        self._min_split = min_split
        self._min_gain = min_gain
        self._root = None

        self.features = None

    def fit(self, X, Y) -> None:
        X, Y = np.atleast_2d(np.array(X)), np.array(Y).flatten()
        self.features = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))

        self._root = self._build_tree(X, Y, depth=0, available_features=list(range(X.shape[1])))
    
    def predict(self, X):
        if self._root is None:
            raise RuntimeError("Cannot predict with an unfitted model")

        X = np.atleast_2d(np.array(X))
        ret = []

        for sample in X:
            current = self._root

            while not current.is_leaf():
                cat = sample[current.feature_index]
                if cat not in current.children:
                    decision = current.majority
                    break
                    
                current = current.children[cat]
            else:
                decision = current.decision

            ret.append(decision)
        return ret
    
    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int, available_features: List[int]) -> Dict:
        X_remains = X[:, available_features]
        classes_labels, classes_counts   = np.unique(Y, return_counts=True)
        majority = classes_labels[np.argmax(classes_counts)]
            
        if len(classes_labels) == 1:
            return _TreeNode(
                depth=depth,
                entropy=0,
                samples=(classes_labels, classes_counts),
                majority=majority,
                decision=classes_labels[0],
            )
        
        if X_remains.shape[1] == 0 or depth == self._max_depth or X_remains.shape[0] < self._min_split:
            return _TreeNode(
                depth=depth,
                majority=majority,
                entropy=entropy(Y),
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        infogains = information_gain(Y, X_remains)
        max_IG_index = np.argmax(infogains)
        splitter_index = available_features[max_IG_index]

        if infogains[max_IG_index] < self._min_gain:
            return _TreeNode(
                depth=depth,
                majority=majority,
                entropy=entropy(Y),
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        categories = np.unique(X_remains[:, max_IG_index])
        children = dict()

        new_available_features = [f for f in available_features if f != splitter_index]
        for cat in categories:
            x_subset, y_subset = X[X[:, splitter_index] == cat], Y[X[:, splitter_index] == cat]
            children[cat] = self._build_tree(x_subset, y_subset, depth+1, new_available_features)

        return _TreeNode(
            depth=depth,
            majority=majority,
            entropy=entropy(Y),
            samples=(classes_labels, classes_counts),
            feature=self.features[splitter_index],
            feature_index=splitter_index,
            children=children
        )
    
            
class _TreeNode:
    def __init__(self, depth: int, majority: int | str, entropy: float, samples: tuple[List, List], feature: str | int | None = None, feature_index: int | None = None, decision: str | None = None, children: Dict = None):
        self.depth = depth
        self.feature = feature
        self.feature_index = feature_index
        self.entropy = entropy
        self.samples = samples
        self.majority = majority
        self.decision = decision
        self.children = children if children is not None else {}

    def is_leaf(self) -> bool:
        return self.decision is not None