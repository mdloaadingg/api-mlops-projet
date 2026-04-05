# Projet MLOps - Prédiction Immobilière (M1 IA)

Ce dépôt contient l'intégralité de mon projet de validation pour le module MLOps. 
L'objectif principal ici n'était pas de chercher l'accuracy absolue avec un modèle surpuissant, mais de concevoir une **pipeline complète et industrialisable** : de l'entraînement tracé jusqu'au déploiement d'une API REST automatisée.

## 1. Le Dataset et le Cas d'Usage
J'ai choisi de travailler sur un jeu de données immobilier (`immo_data.csv`).
* **La tâche :** Classification (Prédire si un bien sera vendu rapidement ou non, variable `sold`).
* **Pourquoi ce choix ?** L'immobilier est un cas d'usage très concret. Il m'a permis de travailler sur des variables mixtes (numériques comme la `surface` et catégorielles comme la `city`) nécessitant la mise en place d'un `ColumnTransformer` propre dans scikit-learn.
* **Taille :** Volontairement réduit pour permettre des entraînements rapides et réguliers dans la boucle CI/CD.

## 2. Architecture et Stack Technique
* **Modélisation :** `scikit-learn` (RandomForestClassifier, Regréssion Logistique).
* **Tracking :** `MLflow` (pour la comparaison des hyperparamètres et la sauvegarde des artefacts).
* **API :** `FastAPI` + `Uvicorn` (avec validation des données via Pydantic).
* **Tests :** `Pytest` (couverture des routes de l'API et gestion des erreurs 422/503).
* **Déploiement :** Docker & Render.

## 3. Comment lancer le projet en local ?

**1. Activer l'environnement virtuel :**
```bash
python -m venv venv
# Sur Windows :
.\venv\Scripts\activate

**2. Installer les dépendances :**
pip install -r requirements.txt

**3. Entraîner le modèle et lancer MLflow :**
python src/mlops_tp/train.py
mlflow ui --port 5001

**4. Lancer lAPI locale :**
uvicorn src.mlops_tp.api:app --reload

4. Défis techniques rencontrés & Apprentissages
Au cours de ce projet, j'ai fait face à plusieurs problématiques intéressantes qui m'ont forcé à revoir ma copie :

Le piège des terminaux Windows : J'ai eu pas mal de conflits de ports ([WinError 10022]) avec Uvicorn et MLflow qui tournaient en fond sans se fermer proprement. J'ai dû adapter mes scripts pour forcer le port 5001 sur MLflow.

Context Manager Pytest : Au début, mes tests d'API échouaient avec une erreur 503 car le modèle ne se chargeait pas pendant le test. J'ai appris à utiliser le bloc with TestClient(app) pour simuler le démarrage complet du serveur (@app.on_event("startup")).

Data Leakage : J'ai fait le choix d'encapsuler tout mon preprocessing (OneHotEncoder, StandardScaler) directement dans un Pipeline Scikit-Learn avec le modèle. Cela me garantit que l'API applique exactement les mêmes transformations qu'à l'entraînement, sans risque d'erreur.
