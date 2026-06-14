import numpy as np
from pandas import DataFrame

from typing import Dict, List
from numpy.typing import ArrayLike

from core.base import BaseEstimator, ClassifierScoreMixin
from core.utils import entropy, information_gain

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

#ID3 - only base desion tree for the moment, only categorical features.
class DecisionTreeClassifier(BaseEstimator, ClassifierScoreMixin):
    def __init__(self, max_depth: int | None = None, min_split: int = 2, min_gain: float = 1e-4) -> None:
        super().__init__()

        self.max_depth = max_depth
        self.min_split = min_split
        self.min_gain  = min_gain

        self._features = None
        self._classes  = None
        self._root     = None

    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()

        self._classes  = np.unique(y)
        self._features = X.columns if isinstance(X, DataFrame) else list(range(x.shape[1]))
        self._root     = self._build_tree(x, y, depth=0, available_features=list(range(x.shape[1])))
    
    def predict(self, X: ArrayLike):
        if self._root is None:
            raise RuntimeError("Cannot predict with an unfitted model")

        x = np.atleast_2d(np.array(X))
        ret = []

        for sample in x:
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

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("Cannot predict_proba with an unfitted model")

        x = np.atleast_2d(np.asarray(X))
        probs = np.zeros((x.shape[0], self._classes.size), dtype=float)

        for i, sample in enumerate(x):
            counts = self._traverse(sample, self._root)
            probs[i] = counts / np.sum(counts)

        return probs

    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int, available_features: List[int]) -> _TreeNode:
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
        
        if X_remains.shape[1] == 0 or depth == self.max_depth or X_remains.shape[0] < self.min_split:
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

        if infogains[max_IG_index] < self.min_gain:
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
            feature=self._features[splitter_index],
            feature_index=splitter_index,
            children=children
        )
    
    def _traverse(self, sample: np.ndarray, node: _TreeNode) -> np.ndarray:
        if node.is_leaf():
            return self._get_node_counts(node)

        cat = sample[node.feature_index]
        if cat not in node.children:
            return self._get_node_counts(node)

        return self._traverse(sample, node.children[cat])

    def _get_node_counts(self, node: _TreeNode) -> np.ndarray:
        labels, counts = node.samples
        full_counts = np.zeros(self._classes.size, dtype=float)
        
        indices = np.searchsorted(self._classes, labels)

        safe_indices = np.minimum(indices, len(self._classes) - 1)
        mask = self._classes[safe_indices] == labels

        safe_indices = safe_indices[mask]
        safe_counts = counts[mask]

        
        full_counts[safe_indices] = safe_counts
        return full_counts