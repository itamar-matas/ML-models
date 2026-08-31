import numpy as np
from pandas import DataFrame

from typing import Dict, List, Callable
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike

from core.base import BaseEstimator, ClassifierScoreMixin
from core.utils.impurity import *

class _TreeNode:
    def __init__(self, depth: int, majority: int | str, samples: tuple[List, List], feature_index: int | None = None, decision: str | None = None, children: Dict = None):
        self.depth = depth
        self.majority = majority
        self.samples = samples

        self.feature_index = feature_index
        self.decision = decision
        self.children = children if children is not None else {}

    def is_leaf(self) -> bool:
        return self.decision is not None

class _BinTreeNode(_TreeNode):
    def __init__(self, depth: int, majority: int | str, samples: tuple[List, List], feature_index: int | None = None, decision: str | None = None, children: Dict | None = None, threshold: float | None = None):
        super().__init__(depth, majority, samples, feature_index, decision, children or {"left": None, "right": None})
        self.threshold = threshold

    @property
    def right(self):
        return self.children.get('right')
    
    @property
    def left(self):
        return self.children.get('left')
    
    @right.setter
    def right(self, node):
        self.children['right'] = node
    
    @left.setter
    def left(self, node):
        self.children['left'] = node


class BaseDecisionTreeClassifier(BaseEstimator, ClassifierScoreMixin):
    def __init__(self, max_depth: int | None = None, min_split: int = 2, min_gain: float = 0.0, criterion: str = "gini") -> None:
        super().__init__()

        self.max_depth = max_depth
        self.min_split = min_split
        self.min_gain  = min_gain

        self._criterion = criterion
        self._features = None
        self._classes  = None
        self._root     = None

    def fit(self, X: ArrayLike, Y: ArrayLike) -> None:
        x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()
        g, m = (gini_gain, gini_method) if self._criterion == "gini" else (information_gain, entropy_method)

        self._classes  = np.unique(y)
        self._features = X.columns if isinstance(X, DataFrame) else list(range(x.shape[1]))
        self._root     = self._build_tree(x, y, depth=0, available_features=list(range(x.shape[1])), gain=g, method=m)

    def predict(self, X: ArrayLike):
        if self._root is None:
            raise RuntimeError("Cannot predict with an unfitted model")

        x = np.atleast_2d(np.array(X))
        ret = []

        for sample in x:
            node = self._traverse(sample, self._root)
            ret.append(node.decision if node.is_leaf() else node.majority)
            
        return ret
    
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("Cannot predict_proba with an unfitted model")

        x = np.atleast_2d(np.asarray(X))
        probs = np.zeros((x.shape[0], self._classes.size), dtype=float)

        for i, sample in enumerate(x):
            node = self._traverse(sample, self._root)
            classes, counts = node.samples
            
            indices = np.searchsorted(self._classes, classes)
        
            padcounts = np.zeros(self._classes.size, dtype=float)
            padcounts[indices] = counts

            probs[i] = padcounts / np.sum(counts)

        return probs

    @abstractmethod
    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int, available_features: List[int] | None = None, gain: Callable | None = None, method: Callable | None = None) -> _TreeNode:
        pass

    @abstractmethod
    def _traverse(self, sample: np.ndarray, node: _TreeNode) -> _TreeNode:
        pass

# ID3 - multinomial branched desion tree for categorical features.
class ID3_DecisionTreeClassifier(BaseDecisionTreeClassifier):
    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int, available_features: List[int] | None = None, gain: Callable | None = None, method: Callable | None = None) -> _TreeNode:
        X_remains = X[:, available_features]
        classes_labels, classes_counts   = np.unique(Y, return_counts=True)
        majority = classes_labels[np.argmax(classes_counts)]
            
        if len(classes_labels) == 1:
            return _TreeNode(
                depth=depth,
                samples=(classes_labels, classes_counts),
                majority=majority,
                decision=classes_labels[0],
            )
        
        if X_remains.shape[1] == 0 or depth == self.max_depth or X_remains.shape[0] < self.min_split:
            return _TreeNode(
                depth=depth,
                majority=majority,
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        gains = gain(Y, X_remains)
        max_gain_index = np.argmax(gains)
        splitter_index = available_features[max_gain_index]

        if gains[max_gain_index] < self.min_gain:
            return _TreeNode(
                depth=depth,
                majority=majority,
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        categories = np.unique(X_remains[:, max_gain_index])
        children = dict()

        new_available_features = [f for f in available_features if f != splitter_index]
        for cat in categories:
            x_subset, y_subset = X[X[:, splitter_index] == cat], Y[X[:, splitter_index] == cat]
            children[cat] = self._build_tree(x_subset, y_subset, depth+1, new_available_features, gain=gain, method=method)

        return _TreeNode(
            depth=depth,
            majority=majority,
            samples=(classes_labels, classes_counts),
            feature_index=splitter_index,
            children=children
        )
    
    def _traverse(self, sample: np.ndarray, node: _TreeNode) -> _TreeNode:
        if node.is_leaf():
            return node

        cat = sample[node.feature_index]
        if cat not in node.children:
            return node

        return self._traverse(sample, node.children[cat])
    
# CART - binary branched decision tree for continuous features.
class CART_DecisionTreeClassifier(BaseDecisionTreeClassifier):
    def _build_tree(self, X: np.ndarray, Y: np.ndarray, depth: int, available_features: List[int] | None = None, gain: Callable | None = None, method: Callable | None = None) -> _TreeNode:
        X_remains = X[:, available_features]
        classes_labels, classes_counts = np.unique(Y, return_counts=True)
        majority = classes_labels[np.argmax(classes_counts)]
            
        if len(classes_labels) == 1:
            return _BinTreeNode(
                depth=depth,
                samples=(classes_labels, classes_counts),
                majority=majority,
                decision=classes_labels[0],
            )
        
        if X_remains.shape[1] == 0 or depth == self.max_depth or X_remains.shape[0] < self.min_split:
            return _BinTreeNode(
                depth=depth,
                majority=majority,
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        thresholds, gains = best_threshold_and_gain(Y, X_remains, method)
        splitter_index = np.argmax(gains)

        max_threshold = thresholds[splitter_index]

        if gains[splitter_index] < self.min_gain:
            return _BinTreeNode(
                depth=depth,
                majority=majority,
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        left_indices = X_remains[:, splitter_index] <= max_threshold
        if not np.any(left_indices) or np.all(left_indices):
            return _BinTreeNode(
                depth=depth,
                majority=majority,
                samples=(classes_labels, classes_counts),
                decision=majority
            )

        children = {
            "right": self._build_tree(X[~left_indices], Y[~left_indices], depth+1, available_features, gain=gain, method=method),
            "left": self._build_tree(X[left_indices], Y[left_indices], depth+1, available_features, gain=gain, method=method)
        }

        return _BinTreeNode(
            depth=depth,
            majority=majority,
            samples=(classes_labels, classes_counts),
            feature_index=available_features[splitter_index],
            threshold=max_threshold,
            children=children
        )

    def _traverse(self, sample: np.ndarray, node: _BinTreeNode) -> _BinTreeNode:
        if node.is_leaf():
            return node

        next_node = node.left if sample[node.feature_index] <= node.threshold else node.right
        if next_node is None:
            return node

        return self._traverse(sample, next_node)
