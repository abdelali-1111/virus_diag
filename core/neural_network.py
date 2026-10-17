"""Petit réseau de neurones (1 couche cachée) avec numpy."""
import numpy as np
from core.model import BaseModel
from core.optimizer import GradientDescent
from core.logistic_regression import sigmoid


class NeuralNetwork(BaseModel):
    def __init__(self, hidden_size=8, optimizer=None, epochs=800, seed=0):
        self.hidden_size = hidden_size
        self.optimizer = optimizer or GradientDescent(0.1)
        self.epochs = epochs
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None

    def _init(self, d):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.5, (d, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0, 0.5, (self.hidden_size, 1))
        self.b2 = np.zeros(1)

    def _forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        out = sigmoid(h @ self.W2 + self.b2)
        return h, out

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        n, d = X.shape
        self._init(d)
        for _ in range(self.epochs):
            h, out = self._forward(X)
            d_out = (out - y) / n
            gW2 = h.T @ d_out
            gb2 = d_out.sum(axis=0)
            d_h = (d_out @ self.W2.T) * (1 - h ** 2)
            gW1 = X.T @ d_h
            gb1 = d_h.sum(axis=0)
            self.optimizer.step([self.W1, self.b1, self.W2, self.b2],
                                [gW1, gb1, gW2, gb2])
        return self

    def predict(self, X):
        _, out = self._forward(np.asarray(X, dtype=float))
        return out.ravel()
