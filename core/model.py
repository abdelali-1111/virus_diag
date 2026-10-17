"""Classe de base abstraite : tout modèle IA du projet en hérite."""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Entraîne le modèle."""

    @abstractmethod
    def predict(self, X):
        """Retourne les probabilités (entre 0 et 1) pour chaque ligne de X."""
