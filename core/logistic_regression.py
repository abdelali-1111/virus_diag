"""Régression logistique implémentée avec numpy."""
import numpy as np
from core.model import BaseModel
from core.optimizer import GradientDescent


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(BaseModel):
    def __init__(self, optimizer=None, epochs=500):
        self.optimizer = optimizer or GradientDescent(0.1)
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = np.zeros(1)
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            grad_w = X.T @ (p - y) / n
            grad_b = np.array([np.mean(p - y)])
            self.optimizer.step([self.w, self.b], [grad_w, grad_b])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return sigmoid(X @ self.w + self.b)
