## Analyse des résultats observés

L'entraînement a utilisé 15 époques et 60000 images MNIST sur cuda. Les pertes moyennes de la dernière époque sont D=1.1302 et G=1.0139. Ces deux fonctions ont des échelles et des objectifs différents : leur comparaison directe ne mesure pas une exactitude de classification.

La variation absolue moyenne des pixels à style fixe et bruit variable est 0.003342 sur l'échelle [-1,1]. Cette mesure confirme un effet numérique du bruit ; sa portée perceptuelle dépend des grilles. La matrice de mélange utilise le style de ligne au premier bloc 7×7 et le style de colonne aux blocs 14×14 et 28×28.

La troncature ramène W vers sa moyenne et réduit la variation des styles. L'interpolation en Z traverse le mapping non linéaire, tandis que celle en W relie directement les styles. La projection t-SNE de 500 points n'est pas étiquetée par classe : elle ne démontre aucune séparation sémantique des chiffres.

La dernière grille du second essai contient plusieurs formes différentes, dont des 0 et des 9 reconnaissables, ainsi que des formes ambiguës. Pour la paire illustrée, le passage en W semble plus progressif que celui en Z ; cette observation qualitative ne vaut que pour cet exemple. Dans la matrice de mélange, plusieurs cases hors diagonale sont fragmentées : le modèle n'a donc pas appris une séparation fiable entre styles grossiers et fins. La troncature vers la moyenne change même la catégorie apparente du chiffre ; elle ne garantit pas une meilleure qualité pour chaque valeur de psi. À code fixé, le bruit laisse globalement la forme stable et produit ici de petites variations de contour/intensité.

Le premier essai de 15 époques s'est effondré vers une forme presque unique ; ses sorties sont conservées dans `premier_essai_effondrement/`. Une seconde exécution de 15 époques ajoute une statistique d'écart-type du batch au discriminateur, suivie d'une convolution. Les figures principales correspondent à cette seconde exécution. Cette comparaison de deux entraînements ne permet pas d'attribuer quantitativement tout écart à cette seule modification.

Les grilles successives permettent d'inspecter l'apparition de structures et les défauts conservés. Le modèle pédagogique possède seulement trois blocs de synthèse et n'utilise ni convolutions modulées/démodulées ni régularisation de longueur de chemin de StyleGAN2. Un entraînement de 15 époques ne suffit pas à revendiquer la qualité d'un modèle haute résolution.
