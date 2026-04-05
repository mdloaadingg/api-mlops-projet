from pydantic import BaseModel, Field
from typing import Dict

# --- SCHÉMAS DE VALIDATION (Pydantic) ---
# C'est ici qu'on définit le "contrat" de notre API. 
# FastAPI va utiliser ces classes pour bloquer automatiquement les mauvaises requêtes 
# et renvoyer une belle Erreur 422 sans faire planter le serveur.

class HouseFeatures(BaseModel):
    # FIXME (Résolu) : J'ai remplacé le paramètre 'example' par 'json_schema_extra' 
    # pour faire disparaître les gros warnings jaunes de Pydantic V2 lors des tests Pytest !
    surface: float = Field(..., description="Surface en mètres carrés", json_schema_extra={"example": 52.0})
    rooms: int = Field(..., description="Nombre de pièces", json_schema_extra={"example": 2})
    city: str = Field(..., description="Ville du bien", json_schema_extra={"example": "Clermont-Ferrand"})

# Structure de la requête entrante (Imposée par le contrat d'interface du TP)
class PredictRequest(BaseModel):
    features: HouseFeatures

# Structure de la réponse renvoyée à l'utilisateur
class PredictResponse(BaseModel):
    prediction: str
    task: str
    
    # On utilise un dictionnaire dynamique pour les probas pour ne pas casser 
    # l'API si on décide de changer les noms des classes ("yes"/"no") à l'avenir.
    proba: Dict[str, float] 
    
    model_version: str
    latency_ms: float