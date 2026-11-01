# MLOps: from notebook to deployed model

 prendre le CNN du projet 2 (digits classifier) et le déployer proprement : tracking d'expériences, API de serving, containerisation.

## Objectif

Passer de "un script qui imprime un résultat" à "un modèle qui tourne en continu et qu'on peut interroger" — la compétence qui différencie un ingénieur ML d'un data scientist pur.

## Architecture

```
train_tracked.py  →  MLflow (tracking + registre des runs)
                            ↓
                     models/best_model.pt + best_model.json (meilleur run, versionné dans Git)
                            ↓
serve.py (FastAPI)   →  charge le modèle, expose /predict
                            ↓
Dockerfile           →  containerise l'ensemble
```

## Ce que fait chaque partie

**`train_tracked.py`** — Entraîne le CNN avec 3 configurations d'hyperparamètres différentes (graine fixée à 42), en enregistrant pour chaque run : les hyperparamètres, la loss à chaque epoch, l'accuracy finale, et le modèle lui-même (state_dict). À la fin, le meilleur run est exporté dans `models/` (poids + métadonnées : run_id MLflow, accuracy, hyperparamètres).

**`serve.py`** — API FastAPI qui charge `models/best_model.pt` (portable : fonctionne après un simple `git clone`, et dans Docker). Les métadonnées du run (run_id MLflow, accuracy) sont renvoyées par `GET /` : on sait exactement quelle version tourne. Pour charger directement un autre run depuis MLflow : `MODEL_RUN_ID=<run_id> uvicorn serve:app`.
- `GET /` — health check (modèle chargé, run_id, accuracy)
- `POST /predict` — reçoit 64 pixels (image 8x8), retourne le chiffre prédit + confiance
- `GET /docs` — documentation Swagger générée automatiquement par FastAPI

**`Dockerfile`** — Containerise l'API (code + `models/`) pour un déploiement portable.

## Résultats des 3 runs trackés (graine 42)

| Run | Epochs | LR | Test Accuracy |
|-----|--------|-----|----------------|
| 1 (baseline) | 10 | 0.001 | 0.928 |
| 2 | 20 | 0.001 | 0.967 |
| 3 | 15 | 0.003 | **0.983** |

Le run 3 (meilleure accuracy) est exporté dans `models/` et servi par l'API. Les résultats sont reproductibles à graine fixée ; ils peuvent varier légèrement d'une machine à l'autre.

## Lancer le projet

```bash
pip install -r requirements.txt

# 1. Entraîner, tracker les runs et exporter le meilleur modèle dans models/
cd src
python train_tracked.py

# 2. Visualiser les runs (optionnel)
mlflow ui --backend-store-uri sqlite:///../mlflow.db
# ouvre http://localhost:5000

# 3. Lancer l'API (le modèle models/best_model.pt est déjà fourni)
uvicorn serve:app --reload --port 8000
# ouvre http://localhost:8000/docs pour tester interactivement

# 4. Containeriser (nécessite Docker, depuis la racine du projet)
cd ..
docker build -t digits-classifier-api .
docker run -p 8000:8000 digits-classifier-api
```

## Prochaines améliorations possibles

- Déployer le container sur un service cloud (Railway, Render, AWS ECS)
- Ajouter un test de charge (locust) pour mesurer la latence sous trafic
- CI/CD : réentraîner et redéployer automatiquement via GitHub Actions
- Monitoring de dérive du modèle (data drift) en production
