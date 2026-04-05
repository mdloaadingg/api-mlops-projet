import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import warnings

# On ignore les warnings un peu pénibles de scikit-learn en local
warnings.filterwarnings("ignore")

def run_ml_experiment(n_trees, run_name):
    # ATTENTION : J'ai forcé le port 5001 ici car le 5000 plante souvent sur Windows
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    
    # 1. Définir le nom de l'expérience principale
    mlflow.set_experiment("Analyse_Immobiliere_MLOps")

    with mlflow.start_run(run_name=run_name):
        
        # --- Ajout de mes tags persos pour la traçabilité ---
        mlflow.set_tag("developer", "Votre_Prenom") # TODO: Remplacer par votre prénom
        mlflow.set_tag("dataset_version", "v1.0-fake-data")

        # Chargement des données avec une petite sécurité
        try:
            df = pd.read_csv("data/immo_data.csv")
        except FileNotFoundError:
            print("❌ Erreur : Le fichier immo_data.csv est introuvable. Lancez generate_dataset.py d'abord !")
            return

        X = df[['surface', 'rooms', 'city']]
        y = df['sold']
        
        # Split classique 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. LOG PARAMETERS : Enregistrer la configuration pour pouvoir la reproduire
        mlflow.log_param("nb_arbres", n_trees)
        mlflow.log_param("type_modele", "RandomForest")
        mlflow.log_param("split_test_size", 0.2)

        # Construction du Pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['surface', 'rooms']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['city'])
        ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1))
        ])

        # Entraînement
        pipeline.fit(X_train, y_train)

        # 3. LOG METRICS : Enregistrer les scores
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label="yes")
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 4. LOG MODEL : Sauvegarde automatique de l'environnement
        mlflow.sklearn.log_model(pipeline, "model_immobilier")
        
        # 5. SAUVEGARDE PHYSIQUE : Pour que l'API et Pytest trouvent le fichier
        os.makedirs("src/mlops_tp/artifacts", exist_ok=True)
        joblib.dump(pipeline, "src/mlops_tp/artifacts/model.joblib")
        
        print(f"✅ Run '{run_name}' OK | Arbres: {n_trees} -> Accuracy: {acc:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    print("Démarrage des entraînements... (Vérifiez que le serveur MLflow tourne sur le port 5001)")
    
    # On teste 3 configs différentes pour voir l'impact de la taille de la forêt
    run_ml_experiment(n_trees=10, run_name="Petit_Modele_Test")
    run_ml_experiment(n_trees=100, run_name="Modele_Standard_Prod")
    run_ml_experiment(n_trees=300, run_name="Gros_Modele_Overkill")
    
    print("🚀 Terminé ! Allez voir les graphiques de comparaison sur http://127.0.0.1:5001")