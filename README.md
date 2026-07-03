# Handwritten Digit Classifier

Un classifieur de chiffres manuscrits (0-9) basé sur un CNN (réseau de neurones convolutionnel), entraîné avec PyTorch sur un dataset Kaggle.

## Utilisation de l'IA

Ce README ainsi que les messages affichés dans la console (print statements) ont été générés avec l'aide d'une IA. Tout le reste — architecture du modèle, logique d'entraînement et d'évaluation, gestion des données, debug — a été écrit par moi.

## Statut actuel (V2)

Le CNN reconnaît correctement la quasi-totalité des chiffres manuscrits testés à la main, à l'exception du 4 — dont le style d'écriture diffère significativement des exemples présents dans le dataset d'entraînement. Le modèle a été entraîné sur un nouveau dataset (PNG à fond transparent, converti en fond blanc au chargement) avec augmentation de données (crop, rotation, variation de contraste/luminosité).

## Fonctionnement du code

- `data.py` : charge le dataset avec `ImageFolder` et un loader custom (`rgba_loader`) qui convertit les PNG à fond transparent en fond blanc. Applique les transformations (redimensionnement, niveaux de gris, normalisation) avec augmentation optionnelle (crop aléatoire, rotation légère, variation de contraste/luminosité).
- `model.py` : définit l'architecture du CNN (deux blocs convolution + activation ReLU + pooling, suivis d'une couche entièrement connectée qui produit les 10 scores de classification).
- `engine.py` : contient la boucle d'entraînement (`train`), l'évaluation par batch sur le set de test (`test`), et le test sur une image unique (`test_one`).
- `main.py` : point d'entrée en ligne de commande (via `click`), qui sélectionne le mode d'exécution et gère le chargement/la sauvegarde des poids du modèle (`model.pth`, sauvegardés uniquement quand la précision s'améliore).

## Résultats

| Epochs | Dimension d'entrée | Learning rate | Split train/test | Augmentation de données | Précision (test set) |
|---|---|---|---|---|---|

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

Entraîner le modèle :
```bash
python main.py --mode train
```

Évaluer la précision sur le set de test :
```bash
python main.py --mode test
```

Tester sur une image aléatoire du dataset :
```bash
python main.py --mode test_one
```

Tester sur une image externe :
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

- Résoudre la confusion sur le chiffre 4 (style d'écriture non représenté dans le dataset)
- Be able to run on gpu on google colab
- Compléter le tableau de résultats avec les différentes combinaisons d'hyperparamètres
- Programmation propre et sécurisée : typage, docstrings, gestion des erreurs
- Data augmentation offline si nécessaire
- Traduction en anglais