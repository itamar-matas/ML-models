import numpy as np
from core import GD

from numpy.typing import ArrayLike
from core import BaseEstimator, ClassifierScoreMixin

_eps = 1e-10

class LogisticRegressionClassifier(BaseEstimator, ClassifierScoreMixin):
    def __init__(self, learning_rate: float = 0.01, C: float = 1.0):
        self.learning_rate = learning_rate
        self.C             = C
        self._theta        = None

        self._classes      = None

    def fit(self, X: ArrayLike, Y: ArrayLike, opt: str = 'gd', multi_class: str = 'multinomial', penalty: str = 'L2', **kwargs):
        x, y = np.atleast_2d(np.asarray(X)), np.asarray(Y).ravel()
        self._classes = np.unique(y)
        n = y.size

        self._theta = np.random.uniform(-1, 1, size=(self._classes.size, x.shape[1] + 1))
        optimization = GD(self._theta.copy(), self.learning_rate, **kwargs)

        def loss_fn(theta):
            W = theta[:, 1:]
            b = theta[:, 0]

            logits = x @ W.T + b
            logits -= np.max(logits, axis=1, keepdims=True)

            exp_logits = np.exp(logits)
            P = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + _eps)

            correct_logprobs = -np.log(P[np.arange(n), y] + _eps)
            data_loss = np.mean(correct_logprobs)

            reg_loss = (1 / (self.C * n)) * np.sum(theta[:, 1:] ** 2)

            return data_loss + reg_loss

        def grad_fn(theta):
            W = theta[:, 1:]
            b = theta[:, 0]

            logits = x @ W.T + b
            logits -= np.max(logits, axis=1, keepdims=True)

            exp_logits = np.exp(logits)
            P = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + _eps)

            E = P.copy()
            E[np.arange(n), y] -= 1
            E /= n

            dW = E.T @ x + (2 / (self.C * n)) * theta[:, 1:]
            db = np.sum(P, axis=0)

            return np.hstack([db.reshape(-1, 1), dW])

        self._theta = optimization.run(loss=loss_fn, gradient=grad_fn)

    def predict(self, X: ArrayLike):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X: ArrayLike): 
        if self._theta is None:
            raise ValueError("cannot predict probabilities with an unfitted model")
        
        x = np.atleast_2d(np.asarray(X))

        logits = np.dot(x, self._theta[:,  1:].T) + self._theta[:, 0]
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        return exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + _eps)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))