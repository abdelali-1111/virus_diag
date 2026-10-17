"""Évaluation d'un modèle déjà entraîné."""
import numpy as np
from utils import metrics


class Evaluator:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def evaluate(self, X, y):
        proba = self.model.predict(X)
        y_pred = (np.asarray(proba) >= self.threshold).astype(int)
        return {
            "accuracy": metrics.accuracy(y, y_pred),
            "precision": metrics.precision(y, y_pred),
            "recall": metrics.recall(y, y_pred),
            "f1": metrics.f1_score(y, y_pred),
        }
