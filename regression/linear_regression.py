# Regression linéaire : Essaye de voir comment les données sont regroupées
# sur un repère orthonormé en traçant une ligne entre les points : best fit line

# Cette ligne essaye de minimiser la distance entre elle et tous les points

# L'équation de la "best fit line" est le modèle des données, 
# permettant de faire des prédictions

# Pros : simple, mathématique
# Cons : les données ne seront pas toujours comme ça

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Données d'entraînement (par exemple, une seule variable)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # `reshape` pour transformer en matrice colonne
y = np.array([2, 4, 5, 4, 5])

# Initialiser le modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(X, y)

# Récupérer les coefficients du modèle
m = model.coef_[0]  # Pente
b = model.intercept_  # Ordonnée à l'origine

# Affichage des résultats
print(f"Pente (m): {m}")
print(f"Ordonnée à l'origine (b): {b}")

# Génération des prédictions
y_pred = model.predict(X)

# Visualisation des résultats
plt.scatter(X, y, color="blue", label="Données")
plt.plot(X, y_pred, color="red", label="Régression linéaire")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
