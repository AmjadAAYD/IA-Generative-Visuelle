# TP3 GAN

**Amjad AAYD — [information masquée] — [information masquée]**

Ouvrir [TP3_GAN.ipynb](TP3_GAN.ipynb) pour consulter le code, les explications et les sorties enregistrées. Le script `.py` contient les mêmes cellules, sans les sorties. Le [compte rendu PDF](Compte_Rendu.pdf) rassemble les figures et l'analyse ; les [résultats détaillés](Resultats_et_analyse.md) précisent les limites des mesures.

## Correspondance avec les questions

| Questions | Réponse dans le notebook |
|---|---|
| 1–4 | Cercle, Sequential, minimax, detach et seuil 0,5 |
| 5 | Demi-lunes, extension optionnelle exécutée |
| 6 | Swiss roll et dimensions latentes 1, 2 et 10 |
| 7 | cGAN MNIST, condition one-hot |
| 8 | Architecture convolutionnelle DCGAN |
| 9 | code2image : formes (n,100) et (n,100,1,1) |
| 10 | Histogramme de 1 000 pseudo-classes et référence réelle |
| 11 | SVM réel/réel, réel/synthétique, synthétique/réel ; sous-ensemble réel explicite |
| 12 | Multiplication de la norme de z |
| 13 | Direction de barycentres pseudo-étiquetés |
| 14 | Interpolation linéaire et sphérique : images et normes |

## Reproduction

Suivre l'installation du README à la racine, puis exécuter les cellules dans l'ordre avec le dossier du notebook comme répertoire courant. Les images MNIST sont téléchargées automatiquement dans `../data/`. Les checkpoints sont créés dans `models/`, exclu de Git. Un accélérateur réduit la durée du calcul ; les sorties publiées proviennent de CUDA.

Les grilles sont exportées dans `figures/`. Aucun poids du professeur n'a été fourni : les modèles DCGAN et LeNet5 du TP GAN sont des remplacements locaux explicitement identifiés. Le Mini-StyleGAN est entraîné depuis une initialisation aléatoire.
