"""Interface clinique en ligne de commande.

Lancer depuis la racine du projet :  python -m app.interface_clinique
"""
import os
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "patient_data.csv")


def main():
    dataset = Dataset(CSV)
    for model in (LogisticRegression(), NeuralNetwork()):
        trainer = Trainer(model, dataset)
        trainer.train()
        scores = Evaluator(model).evaluate(trainer.X_test, trainer.y_test)
        print(f"\n== {type(model).__name__} ==")
        print({k: round(v, 2) for k, v in scores.items()})

        # Le MÊME ClinicalPredictor fonctionne avec les deux modèles
        predictor = ClinicalPredictor(model, trainer.scaler, dataset.feature_names)
        patients = [
            {"age": 70, "temperature": 39.2, "cough": 1, "oxygen_saturation": 89, "crp": 85},
            {"age": 25, "temperature": 36.7, "cough": 0, "oxygen_saturation": 98, "crp": 3},
        ]
        for p in patients:
            print(p, "->", predictor.diagnose(p))


if __name__ == "__main__":
    main()
