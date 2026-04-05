import os
import pytest

# --- TESTS DU PIPELINE D'ENTRAÎNEMENT ---
# L'objectif ici n'est pas de tester les maths de Scikit-Learn, 
# mais de s'assurer que notre "usine" produit bien le fichier requis pour le déploiement.

def test_model_artifact_exists():
    """
    Vérifie que l'entraînement génère bien le modèle sérialisé pour l'API.
    C'est LE test critique pour la CI/CD : s'il échoue, on bloque la mise en production !
    """
    # Utilisation stricte de os.path.join pour éviter les crashs de chemins entre mon Windows local et le Linux de GitHub Actions
    model_path = os.path.join("src", "mlops_tp", "artifacts", "model.joblib")
    
    # Message d'erreur personnalisé (ça m'a sauvé pas mal de temps de débogage quand j'oubliais de lancer train.py)
    assert os.path.exists(model_path), f"❌ Le fichier n'a pas été trouvé au chemin : {model_path}. Avez-vous bien lancé train.py en amont ?"
