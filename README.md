# Handwritten Digit Classifier

Un classifieur de chiffres manuscrits (0-9) basé sur un CNN (réseau de neurones convolutionnel), entraîné avec PyTorch sur un dataset Kaggle.

## Utilisation de l'IA

Ce README ainsi que les messages affichés dans la console (print statements) ont été générés avec l'aide d'une IA. L'ensemble du contenu a été relu et validé par l'auteur. Tout le reste — architecture du modèle, logique d'entraînement et d'évaluation, gestion des données, debug — a été écrit par moi.

## Statut actuel (V2)

Le CNN reconnaît correctement la quasi-totalité des chiffres manuscrits testés à la main. Le modèle a été entraîné sur un dataset Kaggle de chiffres manuscrits avec augmentation de données (crop aléatoire, rotation légère, variation de contraste et de luminosité), ce qui améliore significativement la généralisation par rapport à la V1.

*Remarque : Par la quasi-totalité des chiffres manuscrits testés à la main, on sous-entend tout chiffres relativement bien écrit comparé au dataset. En effet, le dataset comprend des représentations de chiffres parfois différentes de celles que le lecteur fait. Par conséquent, il vous est suggéré de comparer votre écriture à celle du dataset lors de tests manuels via `test_one`.*

## Fonctionnement du code

- `data.py` : charge le dataset avec `ImageFolder` et applique les transformations (redimensionnement, niveaux de gris, normalisation) avec augmentation optionnelle (crop aléatoire, rotation légère, variation de contraste/luminosité).
- `model.py` : définit l'architecture du CNN (deux blocs convolution + activation ReLU + pooling, suivis d'une couche entièrement connectée qui produit les 10 scores de classification).
- `engine.py` : contient la boucle d'entraînement (`train`), l'évaluation par batch sur le set de test (`test`), et le test sur une image unique (`test_one`). Les poids ne sont sauvegardés que lorsque la précision s'améliore.
- `main.py` : point d'entrée en ligne de commande (via `click`), qui sélectionne le mode d'exécution et gère le chargement/la sauvegarde des poids du modèle (`model.pth`).
- `const.py` : centralise tous les hyperparamètres (epochs, learning rate, split, dimensions d'image, paramètres d'augmentation).

## Résultats

*À compléter — tableau des runs avec différentes combinaisons d'hyperparamètres en cours de constitution (support GPU sur Google Colab en développement).*

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
python main.py --mode test_one --image adr_img.png
```
*Remarque : un set d'image de test est disponible dans le dossier `test_data/`.*

## Structure du dataset
La structure du dataset est la suivante :
```
dataset/
0/
1/
2/
...
9/
```
Ce qui permet l'utilisation de `ImageFolder` from `torchvision.datasets`.
## Prochaines étapes

- Support GPU sur Google Colab pour accélérer l'entraînement
- Compléter le tableau de résultats avec différentes combinaisons d'hyperparamètres
- Programmation propre et sécurisée : typage, docstrings, gestion des erreurs
- Source dans le REAMDE (motivation du split et de la structure du réseau : MNIST) (lien dataset kaggle)
- Traduction en anglais

## Continuité possible

Ce projet constitue une base extensible vers des problèmes de reconnaissance visuelle plus larges :

- **Lettres manuscrites** : étendre la classification aux 26 lettres de l'alphabet (voire aux deux casses), en s'appuyant sur la même architecture CNN avec un nombre de classes de sortie adapté.
- **Chiffres en conditions réelles** : entraîner le modèle sur des chiffres photographiés dans des contextes variés — numéros de maison, plaques d'immatriculation, affichages numériques, manuscrits sur tableau blanc — pour une reconnaissance robuste hors contexte contrôlé.
- **OCR généraliste** : combiner les deux extensions ci-dessus pour construire un système capable de lire des séquences de caractères (mots, nombres) plutôt qu'un seul caractère isolé.