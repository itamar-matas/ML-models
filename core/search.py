import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, List

from .base import BaseEstimator, ClassifierProtocol, ModelProtocol
from preprocessing import kFolds, train_test_split

from itertools import product
from copy import deepcopy

class GridSearchCV(BaseEstimator):
    """GridSearchCV is an hyperparameter tuning method which searches through a specified parameter grid to find the best combination of hyperparameters for a given estimator."""
    
    def __init__(self, estimator: ModelProtocol, param_grid: Dict[str, list], cv: int = 5, refit: bool = True, scoring: str = "accuracy", calculate_metrics: List[str] | None = None, random_state: int | None = None):
        """
        Parameters:
        - estimator: The base estimator to be tuned. It should be an instance of a class that inherits from BaseEstimator.
        - param_grid: A dictionary with parameters names as keys and lists of parameter settings to try as values. the key format should be "estimator@param_name"
        - cv: The number of cross-validation folds.
        - scoring: The scoring metric to use for evaluating the performance of each parameter combination.
        """

        self._estimator = estimator
        self._params    = param_grid
        self._kfolds    = kFolds(k=cv, random_state=random_state)
        self._refit     = refit
        self._scoring   = scoring
        self._metrics   = calculate_metrics or []

        self.best_score_      = -float('inf')
        self.best_params_     = None
        self.cv_results_      = dict()
        self.best_estimator_  = None

        self.cv_results_["mean_test_score"] = []
        self.cv_results_["std_test_score"]  = []

        for mtr in self._metrics:
            self.cv_results_[f"metric_{mtr}"] = []

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None, **kwargs):
        x, y = np.atleast_2d(np.asarray(X)), np.array(Y).ravel()

        X_train_heavy, X_valid_heavy, Y_train_heavy, Y_valid_heavy = train_test_split(x, y) 

        for param_name, _ in self._params.items():
            self.cv_results_[f"param_{param_name}"] = []

        keys         = self._params.keys()
        combinations = product(*self._params.values())

        for combo in combinations:
            param_set = dict(zip(keys, combo))
            template_estimator = deepcopy(self._estimator)

            for param, value in param_set.items():
                at_idx = param.find('@')
                setattr(template_estimator[param[:at_idx]], param[at_idx+1:], value) if at_idx != -1 else setattr(template_estimator, param, value)
                
                self.cv_results_[f"param_{param}"].append(value)

            scores = np.zeros(shape=(self._kfolds.get_k_splits()))
            for i, (train, valid) in enumerate(self._kfolds.split(X)):
                current_estimator = deepcopy(template_estimator)

                X_train, Y_train = x[train], y[train]
                X_valid, Y_valid = x[valid], y[valid]

                current_estimator.fit(X_train, Y_train, **kwargs)
                scores[i] = current_estimator.score(X_valid, Y_valid, scores=[self._scoring], onlyvalues=True)[0]

            template_estimator.fit(X_train_heavy, Y_train_heavy, **kwargs)

            for mtr in self._metrics:
                self.cv_results_[f"metric_{mtr}"].append(template_estimator.score(X_valid_heavy, Y_valid_heavy, scores=[mtr], onlyvalues=True)[0])

            current_score = scores.mean()

            self.cv_results_["mean_test_score"].append(current_score)
            self.cv_results_["std_test_score"].append(scores.std())

            if self.best_score_ < current_score:
                self.best_params_, self.best_score_ = param_set, current_score

            
        
        self.best_estimator_ = deepcopy(self._estimator)
        for param, value in self.best_params_.items():
            at_idx = param.find('@')
            setattr(self.best_estimator_[param[:at_idx]], param[at_idx+1:], value) if at_idx != -1 else setattr(self.best_estimator_, param, value)

        if self._refit:
            self.best_estimator_.fit(x, y)

    def predict(self, X: ArrayLike):
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X: ArrayLike):
        if not isinstance(self.best_estimator_, ClassifierProtocol):
            raise RuntimeError("Cannot predict probabilities with a non-classifier model")

        return self.best_estimator_.predict_proba(X)
    
    def score(self, X: ArrayLike, Y: ArrayLike | None = None, scores: List[str] | None = None, onlyvalues=False) -> Dict | List:
        return self.best_estimator_.score(X, Y, scores, onlyvalues)


