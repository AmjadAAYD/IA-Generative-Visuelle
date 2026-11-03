## Analyse des résultats observés

Les poids DCGAN et LeNet5 sont des remplacements entraînés localement. LeNet5 atteint 97.93% d'exactitude sur les 10 000 images de test MNIST. Ce score concerne le classifieur sur des images réelles ; il ne mesure pas la qualité du générateur.

| Expérience SVM linéaire | Score observé | Interprétation |
|---|---:|---|
| Réel → réel | 90.50% | Référence : 10 000 images d'apprentissage réelles, test indépendant de 10 000 images |
| Réel → synthétique | 64.20% | Concordance avec les pseudo-étiquettes de LeNet5 |
| Synthétique → réel | 77.61% | Transfert depuis 1 000 images pseudo-étiquetées, sans égalité de taille d'apprentissage |

Les occurrences synthétiques pour les classes 0 à 9 sont : [102, 71, 107, 102, 79, 91, 85, 121, 113, 129]. Les classes les moins et les plus fréquentes ont respectivement 71 et 129 exemples. La distribution n'est donc pas parfaitement uniforme. Une différence de transfert peut provenir de la diversité et des défauts des images synthétiques, des erreurs de pseudo-étiquetage et de la différence de taille des jeux d'apprentissage. Elle ne permet pas d'isoler une seule de ces causes.

Le cercle généré couvre l'anneau avec quelques points intérieurs. Les demi-lunes ne sont pas reconstruites : la génération forme surtout une concentration triangulaire. Le Swiss roll n'est pas correctement reconstruit non plus : les points forment des branches ou des concentrations plutôt que le rouleau attendu, particulièrement avec un latent de dimension 1. Diminuer la dimension latente réduit les degrés de liberté disponibles, sans suffire à prouver une dimension intrinsèque optimale. Les trois essais Swiss roll ne sont pas des répétitions statistiques.

Le cGAN produit des formes de chiffres avec un bruit important ; les lignes ne correspondent pas systématiquement à la classe demandée. Le conditionnement one-hot est implémenté, mais le contrôle de classe n'est pas acquis de manière fiable après cette exécution de 15 époques. Un entraînement plus long, un discriminateur conditionnel plus expressif et une évaluation par classe seraient des pistes d'amélioration, sans garantie de résultat avant expérimentation.

La norme de z contrôle un déplacement hors de la zone typique de la loi d'apprentissage : les grandes amplitudes peuvent dégrader les chiffres. Les directions issues des barycentres sont des tendances statistiques, pas des transformations de classe garanties. Pour deux codes de rayons proches, la slerp limite la baisse de norme au milieu du trajet que montre l'interpolation linéaire ; elle ne garantit pas que chaque image intermédiaire soit lisible.
