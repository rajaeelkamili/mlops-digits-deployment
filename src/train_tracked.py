"""
Projet 5 (partie 1/3) — MLOps : tracking d'expériences avec MLflow.

Différence avec les projets précédents : au lieu de juste imprimer les
résultats dans le terminal (et les oublier), chaque run d'entraînement
est enregistré -- hyperparamètres, métriques, et le modèle lui-même.
C'est ce qui permet de comparer plusieurs runs et de savoir EXACTEMENT
quelle version d'un modèle tourne en production.
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import json
import random
import shutil
from pathlib import Path

SEED = 42
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("digits-classifier")


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


def train_and_log(epochs=15, lr=0.001, batch_size=32):
    # Graine fixée : un même run redonne (à peu près) le même résultat.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    digits = load_digits()
    X = digits.images.reshape(-1, 1, 8, 8).astype(np.float32) / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DEBUT DU RUN MLFLOW -- tout ce qui se passe ici est tracé automatiquement
    with mlflow.start_run():
        # 1. On enregistre les hyperparamètres AVANT l'entraînement
        mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size, "architecture": "SimpleCNN"})

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            # 2. On enregistre la loss à CHAQUE epoch -- permet de voir la courbe après coup
            mlflow.log_metric("train_loss", train_loss, step=epoch)

        model.eval()
        with torch.no_grad():
            predictions = torch.argmax(model(X_test_t), dim=1)
            accuracy = (predictions == y_test_t).float().mean().item()

        # 3. La métrique finale qui nous intéresse pour comparer les runs
        mlflow.log_metric("test_accuracy", accuracy)

        # 4. Le modèle lui-même est sauvegardé comme "artifact" du run.
        # On sauvegarde le state_dict (poids seuls) plutôt que le format
        # "pt2 export" par défaut de mlflow.pytorch.log_model -- plus portable
        # et plus simple à recharger de façon fiable entre versions de PyTorch.
        torch.save(model.state_dict(), "model_state.pt")
        mlflow.log_artifact("model_state.pt")

        run_id = mlflow.active_run().info.run_id
        print(f"Run enregistré : {run_id}")
        print(f"Test accuracy  : {accuracy:.3f}")

    return run_id, accuracy, {"epochs": epochs, "lr": lr, "batch_size": batch_size}


def export_best(runs):
    """Copie le meilleur modèle dans models/ : c'est CE fichier que l'API charge par défaut.

    Les artifacts MLflow gardent des chemins absolus (propres à ta machine) ; un
    dossier models/ versionné dans Git fonctionne partout, y compris dans Docker.
    """
    run_id, accuracy, params = max(runs, key=lambda r: r[1])
    client = mlflow.tracking.MlflowClient()
    src = client.download_artifacts(run_id, "model_state.pt")
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(src, MODELS_DIR / "best_model.pt")
    (MODELS_DIR / "best_model.json").write_text(
        json.dumps({"run_id": run_id, "test_accuracy": round(accuracy, 4), "params": params, "seed": SEED}, indent=2),
        encoding="utf-8",
    )
    return run_id, accuracy


if __name__ == "__main__":
    # On lance plusieurs runs avec des hyperparamètres différents
    # -- exactement ce qu'on ferait pour chercher la meilleure config.
    runs = []
    print("=== Run 1 : baseline ===")
    runs.append(train_and_log(epochs=10, lr=0.001))

    print("\n=== Run 2 : plus d'epochs ===")
    runs.append(train_and_log(epochs=20, lr=0.001))

    print("\n=== Run 3 : learning rate plus élevé ===")
    runs.append(train_and_log(epochs=15, lr=0.003))

    run_id, acc = export_best(runs)
    print(f"\nMeilleur run : {run_id} avec {acc:.3f} accuracy")
    print("Modèle exporté dans models/best_model.pt (chargé par défaut par serve.py).")
    print("Lance `mlflow ui --backend-store-uri sqlite:///../mlflow.db` pour comparer tous les runs visuellement.")
