"""Régression linéaire implémentée avec numpy."""

import numpy as np
from core.model import BaseModel
from core.optimizer import GradientDescent


class LinearRegression(BaseModel):
    def __init__(self, optimizer=None, epochs=500):
        self.optimizer = optimizer or GradientDescent(0.01)
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = np.zeros(1)

        for _ in range(self.epochs):
            preds = X @ self.w + self.b
            error = preds - y
            grad_w = (X.T @ error) / n
            grad_b = np.array([np.mean(error)])
            self.optimizer.step([self.w, self.b], [grad_w, grad_b])

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b