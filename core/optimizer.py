"""Optimiseur : descente de gradient."""


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, params, grads):
        """Met à jour les paramètres (liste de tableaux numpy) en place."""
        for p, g in zip(params, grads):
            p -= self.lr * g
