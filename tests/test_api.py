from fastapi.testclient import TestClient
import pytest
from src.mlops_tp.api import app

# --- TESTS D'INTÉGRATION DE L'API ---
# Ces tests sont cruciaux pour la CI/CD. Ils simulent un utilisateur réel
# qui envoie des requêtes HTTP à notre serveur FastAPI.

def test_health_check():
    """
    Vérifie que l'API démarre correctement et charge le modèle.
    C'est la route interrogée par Render pour savoir si le service est "Alive".
    """
    # L'utilisation du bloc 'with' est indispensable ici (j'ai perdu pas mal de temps là-dessus !).
    # Cela force FastAPI à exécuter les événements de startup (@app.on_event("startup"))
    # et donc à charger le fichier model.joblib avant de répondre.
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200, "La route /health devrait renvoyer 200 OK."
        assert response.json()["status"] == "ok"

def test_predict_success():
    """
    Vérifie qu'une requête parfaitement formatée renvoie bien une prédiction (Code 200).
    """
    # Payload valide respectant le contrat défini dans schemas.py
    payload = {
        "features": {
            "surface": 52.0,
            "rooms": 2,
            "city": "Clermont-Ferrand"
        }
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # Si ça plante ici, c'est souvent un problème de preprocessing dans le Pipeline Scikit-Learn
        assert response.status_code == 200, f"Erreur attendue 200, reçu: {response.text}"
        
        response_data = response.json()
        assert "prediction" in response_data, "La clé 'prediction' est manquante dans la réponse."
        assert "proba" in response_data, "La clé 'proba' est manquante dans la réponse."

def test_predict_invalid_data():
    """
    Vérifie la robustesse de l'API face aux mauvaises entrées utilisateur.
    Pydantic doit intercepter l'erreur avant même que Scikit-Learn ne soit appelé.
    """
    payload = {
        "features": {
            "surface": "Cinquante",  # Erreur volontaire : attendu float, reçu string
            "rooms": 2,
            "city": "Paris"
        }
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        # FastAPI/Pydantic gère ça automatiquement en renvoyant "422 Unprocessable Entity"
        assert response.status_code == 422, "L'API devrait bloquer les mauvaises données avec un code 422."