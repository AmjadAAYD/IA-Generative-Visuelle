# Mini Projet StyleGAN

**Amjad AAYD — [information masquée] — [information masquée]**

Ouvrir [Mini_Projet_StyleGAN.ipynb](Mini_Projet_StyleGAN.ipynb) pour consulter le code, les explications et les sorties enregistrées. Le script `.py` contient les mêmes cellules, sans les sorties. Le [compte rendu PDF](Compte_Rendu.pdf) rassemble les figures et l'analyse ; les [résultats détaillés](Resultats_et_analyse.md) précisent les limites des mesures.

## Correspondance avec les questions

| Questions | Réponse dans le notebook |
|---|---|
| 1–2 | Architecture style-based, Z/W, usages et éthique |
| 3–4 | MPS/CUDA/CPU, MNIST, forme et intervalle des pixels |
| 5–10 | Mapping, AdaIN, bruit, blocs, générateur et discriminateur |
| 11–12 | Optimiseurs séparés, BCE, grilles après chaque époque |
| 13 | Troncature : psi 0 ; 0,3 ; 0,7 ; 1 ; 1,5 |
| 14 | Interpolation en Z et en W, bruit fixé |
| 15 | Matrice de style mixing 4×4 |
| 16 | Code fixé, bruit variable |
| 17 | t-SNE : 500 points et threads limités |
| 18–20 | StyleGAN2, limites, analyse des résultats et conclusion |

## Reproduction

Suivre l'installation du README à la racine, puis exécuter les cellules dans l'ordre avec le dossier du notebook comme répertoire courant. Les images MNIST sont téléchargées automatiquement dans `../data/`. Les checkpoints sont créés dans `models/`, exclu de Git. Un accélérateur réduit la durée du calcul ; les sorties publiées proviennent de CUDA.

Les grilles sont exportées dans `figures/`. Aucun poids du professeur n'a été fourni : les modèles DCGAN et LeNet5 du TP GAN sont des remplacements locaux explicitement identifiés. Le Mini-StyleGAN est entraîné depuis une initialisation aléatoire.
