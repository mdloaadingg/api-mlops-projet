#  API de Prédiction Immobilière & Pipeline CI/CD MLOps

Ce dépôt contient le code source d'une API de machine learning déployée de bout en bout, de l'entraînement du modèle jusqu'à la mise en production automatisée. 

Ce projet a été développé dans le but de mettre en pratique une véritable architecture **MLOps**, en garantissant la fiabilité du code, la reproductibilité des environnements et l'automatisation des déploiements.

##  Description du Projet

L'application expose un modèle de Machine Learning (Random Forest) capable de prédire si un bien immobilier sera vendu ou non, en fonction de sa surface, de son nombre de pièces et de sa ville. 

Le cœur du projet ne réside pas seulement dans le modèle lui-même, mais dans toute l'infrastructure construite autour pour le rendre accessible, robuste et facile à mettre à jour.

##  Stack Technique

* **Machine Learning :** Scikit-Learn, Pandas
* **Tracking d'expérimentations :** MLflow
* **API REST :** FastAPI, Uvicorn
* **Conteneurisation :** Docker
* **CI/CD :** GitHub Actions
* **Hébergement Cloud :** Render

##  Architecture MLOps

### 1. Intégration Continue (CI)
À chaque modification poussée sur la branche `main` (ou via Pull Request), un workflow GitHub Actions se déclenche automatiquement pour :
1. Installer l'environnement Python.
2. Entraîner le modèle et générer le fichier `model.joblib`.
3. Exécuter les tests unitaires avec `pytest` pour s'assurer que l'API ne régresse pas.
4. Construire l'image Docker dans un environnement vierge pour garantir l'absence de problèmes de dépendances.

### 2. Déploiement Continu (CD)
Une fois la CI validée (pipeline vert), la plateforme **Render** détecte la mise à jour, récupère la dernière image Docker et redéploie le conteneur automatiquement (Zero-Downtime Deployment).

##  Défis techniques et Solutions

Au cours du développement, j'ai dû adapter le code pour qu'il fonctionne aussi bien sur ma machine locale que sur les serveurs distants :

* **Conflit MLflow dans la CI :** Par défaut, MLflow cherchait à se connecter à un serveur local sur le port 5001 (`http://127.0.0.1:5001`), ce qui faisait planter les tests sur l'environnement vierge de GitHub Actions. J'ai résolu ce problème en ajoutant une détection de l'environnement (via la variable `GITHUB_ACTIONS`) pour forcer MLflow à utiliser une sauvegarde en fichier local (`file://`) lors de la CI.
* **Gestion des variables d'environnement :** J'ai dynamisé le port d'écoute d'Uvicorn dans le `Dockerfile` (`--port ${PORT:-10000}`) afin qu'il puisse s'adapter aux exigences réseau de la plateforme Cloud sans rien coder en dur.

##  Installation et Utilisation en local

Si vous souhaitez faire tourner ce projet sur votre machine :

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/VOTRE_PSEUDO/api-mlops-projet.git](https://github.com/VOTRE_PSEUDO/api-mlops-projet.git)
   cd api-mlops-projet

**2. Créer un environnement virtuel et installer les dépendances :**
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt

**3. Générer les données et entraîner le modèle : :**
python generate_dataset.py
python src/mlops_tp/train.py

**4. Lancer l'API :**
uvicorn src.mlops_tp.api:app --reload --port 10000

