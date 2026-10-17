"""ClinicalPredictor : utilise un modèle DÉJÀ entraîné pour diagnostiquer.

Réponse à la question 2 : le concept est le POLYMORPHISME, plus précisément
le « duck typing » en Python. ClinicalPredictor ne connaît pas la classe du
modèle : il appelle seulement model.predict(). Peu importe que ce soit une
régression logistique, un réseau de neurones ou autre.

Réponse à la question 3 : séparer prédiction et entraînement respecte le
principe de responsabilité unique (SRP). On peut ré-entraîner ou changer de
modèle sans toucher au code clinique, déployer uniquement le prédicteur
(léger, sans données d'entraînement), tester chaque partie indépendamment
et éviter toute modification accidentelle du modèle en production.
"""
import numpy as np


class ClinicalPredictor:
    def __init__(self, model, scaler=None, feature_names=None, threshold=0.5):
        self.model = model                  # modèle déjà entraîné
        self.scaler = scaler                # même prétraitement qu'à l'entraînement
        self.feature_names = feature_names  # ordre des variables (si dict)
        self.threshold = threshold

    def _to_array(self, patient_data):
        if isinstance(patient_data, dict):
            if self.feature_names is None:
                raise ValueError("feature_names requis pour un patient au format dict")
            patient_data = [patient_data[f] for f in self.feature_names]
        x = np.asarray(patient_data, dtype=float).reshape(1, -1)
        return self.scaler.transform(x) if self.scaler is not None else x

    def diagnose(self, patient_data):
        x = self._to_array(patient_data)
        prediction = float(np.asarray(self.model.predict(x)).ravel()[0])
        return "Infecté" if prediction >= self.threshold else "Sain"
