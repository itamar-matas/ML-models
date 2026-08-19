import numpy as np
from .base import BaseOptimizer

class GD(BaseOptimizer):
    def __init__(self, theta0: np.ndarray, learning_rate: float = 0.01, max_epochs: int = 100, max_loss: float= 0.001, tolerance: float = 0.01):
        self._theta      = theta0
        
        self.lr         = learning_rate
        self.max_epochs = max_epochs
        self.max_loss   = max_loss
        self.tolerance  = tolerance
   
    def step(self, gradient, **kwargs):
        self._theta -= self.lr * np.clip(gradient(self._theta), -10.0, 10.0)

    def run(self, loss, gradient):
        previous_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            self.step(gradient)

            current_loss = loss(self._theta)
            if np.isnan(current_loss) or current_loss < self.max_loss or abs(previous_loss - current_loss) < self.tolerance:
                break
                
            previous_loss = current_loss
            
        return self._theta