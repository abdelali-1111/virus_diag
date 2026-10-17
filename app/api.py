"""API REST (FastAPI) pour le diagnostic.

Lancer depuis la racine du projet :  python main.py
ou :  uvicorn app.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel

from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from pipeline.trainer import Trainer
from pipeline.predictor import ClinicalPredictor

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "patient_data.csv")


class DiagnosisAPI:
    """Logique indépendante de FastAPI : dict -> dict (utilisée aussi par la CLI)."""

    def __init__(self, predictor):
        self.predictor = predictor

    def handle_request(self, payload):
        try:
            return {"status": "ok", "diagnosis": self.predictor.diagnose(payload)}
        except (KeyError, ValueError) as e:
            return {"status": "error", "message": f"Données invalides : {e}"}


def build_predictor(model=None):
    """Entraîne un modèle (Trainer) puis le confie à ClinicalPredictor."""
    dataset = Dataset(CSV)
    trainer = Trainer(model or LogisticRegression(), dataset)
    trainer.train()
    return ClinicalPredictor(trainer.model, trainer.scaler, dataset.feature_names)


class Patient(BaseModel):
    age: float
    temperature: float
    cough: int
    oxygen_saturation: float
    crp: float


app = FastAPI(title="Virus Diagnostic API")
_api = DiagnosisAPI(build_predictor())


@app.get("/")
def root():
    return {"message": "API de diagnostic en ligne. Voir /docs"}


@app.post("/diagnose")
def diagnose(patient: Patient):
    return _api.handle_request(patient.model_dump())
