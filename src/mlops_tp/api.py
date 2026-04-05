from fastapi import FastAPI, HTTPException
import joblib
import os
import time
import pandas as pd
# Import de la structure des données (Pydantic)
from .schemas import PredictRequest, PredictResponse

app = FastAPI(
    title="API MLOps - Prédiction Immobilière",
    description="API permettant d'inférer la rapidité de vente d'un bien (Projet M1 IA)",
    version="0.1.0"
)

# Variables globales (mise en cache du modèle pour éviter de le recharger à chaque requête)
MODEL = None
MODEL_VERSION = "0.1.0"

# --- 1. CHARGEMENT DU MODÈLE EN MÉMOIRE ---
# FIXME: FastAPI remonte un DeprecationWarning pour "on_event". 
# Dans une V2, il faudra passer par un "lifespan context manager", mais je garde ça pour valider le TP car les tests passent.
@app.on_event("startup")
def load_model():
    global MODEL
    # J'utilise os.path.abspath au lieu d'un chemin relatif en dur (ex: "./artifacts/...").
    # C'est indispensable pour que ça ne plante pas lors du déploiement Docker sur Render !
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "artifacts", "model.joblib")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        print(f"✅ Modèle chargé avec succès depuis {model_path}")
    else:
        # On ne crashe pas l'API ici, mais les routes renverront une erreur 503
        print(f"❌ ERREUR CRITIQUE : Modèle introuvable au chemin {model_path}")

# --- 2. LES ROUTES DE L'API ---

@app.get("/health")
def health_check():
    """
    Route vitale pour le Cloud (Render). 
    Si elle renvoie 200, la plateforme sait que le conteneur a bien démarré.
    """
    if MODEL is None:
        # Erreur 503 : Service Unavailable
        raise HTTPException(status_code=503, detail="Modèle non chargé, l'API ne peut pas répondre.")
    return {"status": "ok", "message": "API V2 : La CD fonctionne !"}

@app.get("/metadata")
def get_metadata():
    return {
        "model_version": MODEL_VERSION,
        "task": "classification",
        "expected_features": {
            "surface": "float",
            "rooms": "int",
            "city": "string"
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Reçoit le JSON de l'utilisateur, le valide via Pydantic (gère les erreurs 422 auto), 
    puis interroge le modèle Scikit-Learn.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas disponible.")
    
    start_time = time.time()
    
    try:
        # On reformate les données pour correspondre à l'entrée attendue par le Pipeline Scikit-Learn
        input_data = {
            "surface": request.features.surface,
            "rooms": request.features.rooms,
            "city": request.features.city
        }
        
        # Scikit-Learn est un peu lourd, il exige un DataFrame (2D) même pour une seule prédiction
        df_input = pd.DataFrame([input_data])
        
        # Inférence
        pred = MODEL.predict(df_input)[0]
        proba_array = MODEL.predict_proba(df_input)[0]
        classes = MODEL.classes_
        
        # Mapping dynamique des probabilités (au cas où on change les labels plus tard)
        proba_dict = {str(classes[i]): round(float(proba_array[i]), 2) for i in range(len(classes))}
        
        # Latence en millisecondes
        latency = (time.time() - start_time) * 1000
        
        return PredictResponse(
            prediction=str(pred),
            task="classification",
            proba=proba_dict,
            model_version=MODEL_VERSION,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        # Sécurité ultime au cas où le preprocessing Scikit-Learn plante
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")