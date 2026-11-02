# Image de base légère avec Python
FROM python:3.11-slim

WORKDIR /app

# On copie d'abord requirements.txt seul -- Docker met en cache cette étape
# et ne réinstalle pas les dépendances à chaque changement de code source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Puis le code et le modèle exporté (models/best_model.pt)
COPY src/ ./src/
COPY models/ ./models/

WORKDIR /app/src

EXPOSE 8000

# --host 0.0.0.0 est nécessaire pour que l'API soit accessible depuis
# l'extérieur du container (127.0.0.1 ne suffit pas en Docker).
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
