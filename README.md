# Handwritten Digit Classifier

Un classifieur de chiffres manuscrits (0-9) basé sur un CNN (réseau de neurones convolutionnel), entraîné avec PyTorch sur un dataset Kaggle.

## Utilisation de l'IA

Ce README ainsi que les messages affichés dans la console (print statements) ont été générés avec l'aide d'une IA. Tout le reste — architecture du modèle, logique d'entraînement et d'évaluation, gestion des données, debug — a été écrit par moi.

## Statut actuel (V1)

Le CNN entraîné sur le dataset Kaggle (images 28x28, niveaux de gris) atteint environ 91.8% de précision sur le set de test issu du même dataset, pour 15 epochs. Le modèle généralise cependant mal sur des photos prises à la volée avec la caméra de l'ordinateur : mauvais cadrage, chiffre petit ou excentré dans l'image, fond non uniforme. Les chiffres à traits fins (1, 4, 7) sont plus souvent mal reconnus que ceux à boucles marquées (6, 9), probablement à cause de ce décalage entre les images d'entraînement (propres, centrées) et les photos réelles (bruitées, mal cadrées).

## Fonctionnement du code

- `data.py` : charge le dataset avec `ImageFolder` et applique les transformations (redimensionnement, passage en niveaux de gris, conversion en tenseur, normalisation). Contient aussi `getDataLoader`, qui regroupe les images en batchs mélangés aléatoirement.
- `model.py` : définit l'architecture du CNN (deux blocs convolution + activation ReLU + pooling, suivis d'une couche entièrement connectée qui produit les 10 scores de classification).
- `engine.py` : contient la boucle d'entraînement (`train`), l'évaluation par batch sur le set de test (`test`), et le test sur une image unique (`test_one`).
- `main.py` : point d'entrée en ligne de commande (via `click`), qui sélectionne le mode d'exécution et gère le chargement/la sauvegarde des poids du modèle (`model.pth`). Les poids ne sont sauvegardés que lorsque la précision sur le set de test s'améliore par rapport au meilleur résultat précédent.

## Résultats

| Epochs | Dimension d'entrée | Learning rate | Split train/test | Augmentation de données | Précision (test set) |
|---|---|---|---|---|---|
| 15 | 28x28 | 0.01 | 0.2 | Aucune | 91.8% |
| 30 | 28x28 | 0.01 | 0.2 | Aucune | 94.1% |
| 15 | 32x32 | 0.01 | 0.2 | Aucune | à compléter |
| 30 | 32x32 | 0.01 | 0.2 | Aucune | à compléter |
| 15 | 64x64 | 0.01 | 0.2 | Aucune | 92.3% |
| 30 | 64x64 | 0.01 | 0.2 | Aucune | 92.3% |

D'autres lignes seront ajoutées au fur et à mesure des essais (augmentation de données par crop/rotation/contraste, autres learning rates, autres splits, etc.).

## Installation

Créer et activer un environnement virtuel, puis installer les dépendances listées dans `requirements.txt`.

Sous Windows (PowerShell) :
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS/Linux :
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

Toutes les commandes ci-dessous sont à lancer depuis le dossier `src`, une fois l'environnement virtuel activé.

Entraîner le modèle sur le dataset (sauvegarde automatiquement les meilleurs poids dans `model.pth`) :
```bash
python main.py --mode train
```

Évaluer la précision du modèle déjà entraîné sur l'ensemble du set de test :
```bash
python main.py --mode test
```

Tester le modèle sur une image choisie aléatoirement dans le dataset (affiche la prédiction et compare au vrai label) :
```bash
python main.py --mode test_one
```

Tester le modèle sur une image externe fournie par l'utilisateur (affiche uniquement la prédiction, sans comparaison) :
```bash
python main.py --mode test_one --image suppdata/photo.png
```

## Structure du dataset

Le dataset doit être organisé en un dossier par classe, chaque dossier contenant les images correspondant à ce chiffre :

```
dataset/
0/
1/
2/
...
9/
```

## Prochaines étapes

- Refactoriser pour passer `H`, `W`, `lr`, `split`, `epochs` en paramètres des fonctions plutôt qu'en constantes importées directement, afin de pouvoir automatiser des runs avec différentes combinaisons d'hyperparamètres et compléter le tableau de résultats sans modification manuelle du code à chaque essai
- Data augmentation (crop aléatoire, légère rotation, variation de contraste/luminosité) pour mieux généraliser sur des photos prises en conditions réelles
- Comparer les performances selon la taille d'image d'entrée, le nombre d'epochs, le learning rate et le split train/test
- Programmation propre et sécurisée : typage des paramètres et valeurs de retour dans les fonctions, docstrings décrivant le rôle de chaque fonction, gestion des erreurs (fichier image introuvable, format invalide, etc.)
- Traduire tout en anglais