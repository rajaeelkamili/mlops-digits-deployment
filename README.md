# MLOps: from notebook to deployed model

 prendre le CNN du projet 2 (digits classifier) et le déployer proprement : tracking d'expériences, API de serving, containerisation.

## Objectif

Passer de "un script qui imprime un résultat" à "un modèle qui tourne en continu et qu'on peut interroger" — la compétence qui différencie un ingénieur ML d'un data scientist pur.

## Architecture

```
train_tracked.py  →  MLflow (tracking + registre des runs)
                            ↓
                       model_state.pt (artifact du meilleur run)
                            ↓
serve.py (FastAPI)   →  charge le modèle par run_id, expose /predict
                            ↓
Dockerfile           →  containerise l'ensemble
```

## Ce que fait chaque partie

**`train_tracked.py`** — Entraîne le CNN avec 3 configurations d'hyperparamètres différentes, en enregistrant pour chaque run : les hyperparamètres, la loss à chaque epoch, l'accuracy finale, et le modèle lui-même (state_dict).

**`serve.py`** — API FastAPI qui charge un modèle par son `run_id` MLflow (traçabilité : on sait exactement quelle version tourne) et expose :
- `GET /` — health check
- `POST /predict` — reçoit 64 pixels (image 8x8), retourne le chiffre prédit + confiance
- `GET /docs` — documentation Swagger générée automatiquement par FastAPI

**`Dockerfile`** — Containerise l'API pour un déploiement portable.

## Résultats des 3 runs trackés

| Run | Epochs | LR | Test Accuracy |
|-----|--------|-----|----------------|
| 1 (baseline) | 10 | 0.001 | 0.956 |
| 2 | 20 | 0.001 | 0.953 |
| 3 | 15 | 0.003 | **0.975** |

Run déployé dans `serve.py` : le run 3 (meilleure accuracy).

## Lancer le projet

```bash
pip install -r requirements.txt

# 1. Entraîner et tracker les runs
cd src
python train_tracked.py
# Note le run_id affiché à la fin, mets-le à jour dans serve.py (variable RUN_ID)

# 2. Visualiser les runs (optionnel)
mlflow ui --backend-store-uri sqlite:///../mlflow.db
# ouvre http://localhost:5000

# 3. Lancer l'API
uvicorn serve:app --reload --port 8000
# ouvre http://localhost:8000/docs pour tester interactivement

# 4. Containeriser (nécessite Docker)
cd ..
docker build -t digits-classifier-api .
docker run -p 8000:8000 digits-classifier-api
```

## Prochaines améliorations possibles

- Déployer le container sur un service cloud (Railway, Render, AWS ECS)
- Ajouter un test de charge (locust) pour mesurer la latence sous trafic
- CI/CD : réentraîner et redéployer automatiquement via GitHub Actions
- Monitoring de dérive du modèle (data drift) en production
