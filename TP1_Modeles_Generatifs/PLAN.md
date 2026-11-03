# Introduction aux modèles génératifs — plan

Source : TP_1_introduction_modeles_generatifs.pdf, huit pages lues intégralement.
Binôme repris de cette conversation : Amjad AAYD et [information masquée]. Groupe à compléter.

1. Réutiliser les bibliothèques scientifiques Python 3.12 déjà installées, via un environnement de travail distinct. Calcul CPU, deux threads ; aucune installation CUDA nécessaire.
2. Partie 1 : Bernoulli N=1000 p=.7, gradient direct et sigmoïde ; gaussienne univariée N=2000 mu=2 sigma=1.5, softplus ; gaussienne 3D N=2000, Cholesky ; EM manuel GMM K=3 d=3 N=6000 et alignement ; GMM 2D sinusoïdal N=300, six composantes, génération 200 globale et 100 conditionnelle.
3. Partie 2 : chiffres grayscale load_digits locaux (8x8, 1797 images) ; covariance /N, décomposition spectrale, moyenne/eigen-images ; génération r=10,25,50 ; variation d'une coordonnée latente ; explication G(z)=mu+Az et limites.
4. Partie 3 : corpus français local, tokenisation simple, dictionnaires de bigrammes, dix paires fréquentes, sample_next_word et phrase commençant par Une ; approfondissement ordre 2 ; corpus symbolique exact A/B/C/D/eos, NLL, Laplace, vingt séquences par variante, comparaisons ordre 1/2.
5. Produire trois notebooks exécutés avec figures intégrées, métriques JSON, corpus et sorties, puis un compte rendu français détaillé et des réponses théoriques.
6. Vérification proportionnée : convergence vers MLE, contraintes probabilistes et SPD, EM, shapes/eigen/covariance, probabilités de transition, NLL et génération ; rendu PDF et contenu archive.

Précisions : epsilon~N(0,0.1) est interprété selon la convention du sujet comme une variance de 0.1 (écart-type sqrt(0.1)). Échantillons images affichés dans [0,1] après clipping ; statistiques calculées sur sorties brutes. Graines fixées ; aucun résultat inventé.
