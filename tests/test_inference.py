import joblib
import os
import pandas as pd
import pytest

# --- TESTS D'INFÉRENCE ---
# On vérifie que le modèle ne dit pas n'importe quoi. C'est crucial avant de le brancher 
# à l'API, sinon on risque de renvoyer des prédictions absurdes à l'utilisateur.

def test_prediction_output():
    """Vérifie que le modèle renvoie bien une classe connue et des probabilités logiques."""
    
    model_path = os.path.join("src", "mlops_tp", "artifacts", "model.joblib")
    
    # Si le modèle n'existe pas, on fait échouer le test proprement au lieu d'avoir un crash Python
    if not os.path.exists(model_path):
        pytest.fail(f"Modèle introuvable dans {model_path}. L'entraînement doit passer avant l'inférence.")
        
    model = joblib.load(model_path)
    
    # Données de test (Je prends Lyon car c'est une de nos catégories)
    dummy_data = pd.DataFrame([{"surface": 50.0, "rooms": 2, "city": "Lyon"}])
    
    # Inférence
    pred = model.predict(dummy_data)[0]
    proba = model.predict_proba(dummy_data)[0]
    
    # Vérifications métier
    # FIXME: Si un jour on change "yes/no" en "vendu/non_vendu", il faudra mettre à jour ce test !
    assert pred in ["yes", "no"], f"Prédiction inattendue : {pred}. Attendu: 'yes' ou 'no'."
    assert all(0 <= p <= 1 for p in proba), "Les probabilités de Scikit-Learn sortent de [0, 1], c'est mathématiquement impossible."
    assert round(sum(proba), 5) == 1.0, "La somme des probabilités doit faire 1."