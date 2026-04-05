import pandas as pd
import numpy as np
import os

# Le but est d'avoir un dataset léger pour que l'entraînement prenne < 2 min dans la CI/CD

np.random.seed(42) # On fixe la seed pour garantir la reproductibilité 
n_samples = 500 # Pas besoin d'un million de lignes pour valider l'infrastructure MLOps

cities = ["Clermont-Ferrand", "Lyon", "Paris"]
city_col = np.random.choice(cities, n_samples)
rooms_col = np.random.randint(1, 6, n_samples)
# La surface dépend du nombre de pièces + un peu de bruit gaussien pour le réalisme
surface_col = rooms_col * 15 + np.random.normal(10, 5, n_samples)

# --- Logique métier  ---
# Je crée une règle un peu arbitraire pour que le RandomForest ait un vrai pattern à apprendre
sold_col = []
for i in range(n_samples):
    # Biais assumé : Les grands apparts à Clermont se vendent comme des petits pains !
    if surface_col[i] > 50 and city_col[i] == "Clermont-Ferrand":
        sold_col.append("yes")
    # Les T3+ de plus de 30m² partent vite de manière générale
    elif surface_col[i] > 30 and rooms_col[i] > 2:
        sold_col.append("yes")
    # Le reste met plus de temps à se vendre
    else:
        sold_col.append("no")

df = pd.DataFrame({
    "surface": surface_col,
    "rooms": rooms_col,
    "city": city_col,
    "sold": sold_col # Notre variable cible 
})

# Petite sécurité : je force la création du dossier sinon ça plante au tout premier run
os.makedirs("data", exist_ok=True)

# Sauvegarde
df.to_csv("data/immo_data.csv", index=False)
print(f"✅ Succès : Dataset généré ({n_samples} lignes) dans data/immo_data.csv !")