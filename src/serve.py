"""
Projet 5 (partie 2/3) — MLOps : servir le modèle via une API FastAPI.

Différence avec avoir juste un script train.py : ici le modèle tourne
en continu, prêt à répondre à des requêtes -- c'est ce qui permet à
n'importe quelle application (web, mobile) de l'utiliser.
"""

import json
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Digits Classifier API",
    description="Sert le modèle CNN entraîné pour classifier des chiffres manuscrits (8x8 pixels).",
    version="1.0.0",
)

mlflow.set_tracking_uri("sqlite:///../mlflow.db")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Par défaut, on charge models/best_model.pt (versionné dans Git, donc portable :
# fonctionne après un simple `git clone`, et dans Docker). Les métadonnées
# (run_id MLflow, accuracy) sont dans models/best_model.json : on sait toujours
# quelle version tourne.
# Pour charger un autre run directement depuis MLflow (traçabilité totale) :
#     MODEL_RUN_ID=<run_id> uvicorn serve:app
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_RUN_ID = os.environ.get("MODEL_RUN_ID")

RUN_ID = MODEL_RUN_ID or "unknown"
MODEL_META = {}

model = None  # chargé au démarrage (voir startup event ci-dessous)


@app.on_event("startup")
def load_model():
    global model, RUN_ID, MODEL_META
    if MODEL_RUN_ID:
        local_path = mlflow.tracking.MlflowClient().download_artifacts(MODEL_RUN_ID, "model_state.pt")
    else:
        local_path = MODELS_DIR / "best_model.pt"
        meta_path = MODELS_DIR / "best_model.json"
        if not local_path.exists():
            raise RuntimeError(
                f"{local_path} introuvable : lance d'abord `python train_tracked.py` "
                "(il exporte le meilleur modèle dans models/)."
            )
        if meta_path.exists():
            MODEL_META = json.loads(meta_path.read_text(encoding="utf-8"))
            RUN_ID = MODEL_META.get("run_id", "unknown")
    model = SimpleCNN()
    model.load_state_dict(torch.load(local_path, weights_only=True))
    model.eval()
    print(f"Modèle chargé (run {RUN_ID})")


# Pydantic valide automatiquement la requête -- si le format est incorrect,
# l'API renvoie une erreur claire avant même d'atteindre notre code.
class DigitInput(BaseModel):
    # image 8x8 = 64 pixels, valeurs 0-16 (comme le dataset d'entraînement)
    pixels: List[float] = Field(..., min_length=64, max_length=64,
                                  description="64 valeurs de pixels (grille 8x8), de 0 à 16")


class PredictionOutput(BaseModel):
    predicted_digit: int
    confidence: float
    run_id_used: str


@app.get("/")
def health_check():
    """Endpoint de santé -- vérifie que l'API tourne et que le modèle est chargé."""
    return {"status": "ok", "model_loaded": model is not None, "run_id": RUN_ID, "test_accuracy": MODEL_META.get("test_accuracy")}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: DigitInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    pixels = np.array(input_data.pixels, dtype=np.float32).reshape(1, 1, 8, 8) / 16.0
    tensor_input = torch.tensor(pixels)

    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted].item()

    return PredictionOutput(
        predicted_digit=predicted,
        confidence=round(confidence, 4),
        run_id_used=RUN_ID,
    )


# Lancer avec : uvicorn serve:app --reload --port 8000
# Puis tester sur http://localhost:8000/docs (interface Swagger auto-générée)
