Ce dossier contient les scripts pour la prédiction de trajectoire, qui sont respectivement dans le dossier Mesh et Ocean.
Les résultats sont dans le dossier resultat. Les données sont dant le dossier tp.

### Explication de la structure du code

Pour chaque expérience, il contient 3 scripts python.
- 'main.py' : Ce script est le script principal qui appelle les autres scripts.
- 'DataLoader.py' : Ce script est pour charger les données.
- 'Model.py' : Ce script est pour définir le modèle.
- 'utils.py' : Ce script est pour définir les fonctions utiles, nottament pour le calcul de la performance pour le fold de test ainsi que quelques fonctions plot pour loss, accuracy, matrice de confusion, etc.

### Pour lancer le code
Il suffit de dézipper le dossier tp et de lancer le script main.py. Il est possible de changer les paramètres dans le script main.py. (nommbre de hop, nombre d'ephoch, seuil pour le learly stopping, etc.)