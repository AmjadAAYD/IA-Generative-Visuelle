# Introduction aux modèles génératifs — réponses et résultats

**Binôme : Amjad AAYD et [information masquée] — [information masquée] — 6 octobre 2026**
Enseignant : [information masquée]. Groupe : à compléter.

Les trois notebooks sont exécutés. Les résultats non arrondis sont dans les JSON joints.

## Estimation et génération

Estimer consiste à choisir les paramètres d'une loi expliquant les observations. Générer consiste à tirer de nouvelles observations selon cette loi. Le maximum de vraisemblance maximise la somme des log-probabilités ; la NLL est son opposé. Les résultats du TP sont en nats, avec logarithmes naturels.

## Partie 1

### Bernoulli

N=1000, vrai p=0.7, initialisation θ=0.5.
NLL moyenne = −x̄ log θ − (1−x̄) log(1−θ).
Gradient en θ = (θ−x̄)/(θ(1−θ)). Le MLE est x̄.
La sigmoïde θ=σ(φ) impose 0<θ<1 sans clamp ; son gradient en φ est θ−x̄.

Résultat : MLE=0.696000, gradient direct=0.696000, sigmoïde=0.696000, NLL=0.614215.
L'écart au p vrai vient de la fluctuation de l'échantillon ; les deux optimisations rejoignent le MLE.

### Gaussienne univariée

N=2000, μ vrai=2 et σ vrai=1.5.
NLL moyenne = log σ + 0.5 log(2π) + moyenne((x−μ)²)/(2σ²).
σ=softplus(ρ)+1e−6 garantit la positivité. Le MLE de variance utilise N, pas N−1.

Résultat : μ=1.959693, σ=1.519230, NLL=1.837142. Les estimations concordent avec les références analytiques.

### Gaussienne multivariée

N=2000, d=3. Σ=LLᵀ avec L triangulaire inférieure et diagonale positive.
NLL moyenne = (d/2)log(2π) + somme(log(diag L)) + 0.5 moyenne(||L⁻¹(x−μ)||²).
Résoudre le système triangulaire évite une inversion explicite. Le notebook trace les moyennes, les variances diagonales, la NLL et les corrélations.

Moyenne estimée : [1.0337141426956125, -1.960268127620248, 0.5202039728246989].
Erreur maximale de covariance contre MLE : 4.885e-15.
Les covariances vraies et estimées sont enregistrées intégralement dans metriques_partie1.json.

### GMM et EM

N=6000, K=3, d=3. E-step : calcul des responsabilités normalisées ; M-step : mises à jour pondérées des poids, moyennes et covariances. Le calcul des responsabilités utilise logsumexp. Trois initialisations aléatoires sont comparées ; l'essai de meilleure log-vraisemblance est retenu.

Poids estimés après alignement : [0.2519851291503221, 0.4470148729187404, 0.3009999979309375].
Meilleur essai : 7 mises à jour, log-vraisemblance totale=-30012.693003.
Le label switching ne change pas la loi : les six permutations sont testées et la distance entre moyennes décide de l'alignement. Permutation retenue : [2, 1, 0].
EM peut converger vers un optimum local ; les trois essais ne prouvent pas un maximum global.

### GMM sinusoïdal

300 points, x1 uniforme dans [0,10], x2=sin(x1)+ε. La variance de ε est 0.1 ; son écart-type est sqrt(0.1). GaussianMixture utilise six composantes, des ellipses conditionnelles à 95 %, 200 échantillons globaux et 100 conditionnels sur i=0.

La génération globale choisit une composante selon ses poids ; la génération conditionnelle la fixe. Le mélange a convergé en 9 itérations. Son support non borné peut produire des valeurs hors du domaine du processus original.

## Partie 2

### Données et covariance

Sous-ensemble : 1500 images digits 8×8, normalisées par 16, donc d=64. Moyenne empirique et covariance /N suivent le sujet. Le rang numérique de covariance est 61, car certains pixels sont constants.
La covariance est semi-définie positive ; la construction ne nécessite pas une matrice strictement inversible.

### Générateur affine

G(z)=μ+Az, A=E_r diag(sqrt(λ_r)), z~N(0,I_r).
E[G(z)]=μ et Cov[G(z)]=AAᵀ. G est affine ; seule sa partie Az est linéaire.
Une coordonnée z_j varie selon la direction e_j et une amplitude sqrt(λ_j). Les directions les plus importantes sont celles de plus grande variance, sans signification sémantique garantie.

Variance conservée : r=10 : 73.9196 %, r=25 : 93.3613 %, r=50 : 99.9511 %.
La comparaison utilise les mêmes coordonnées latentes de base. Le clipping sert seulement à l'affichage ; le fichier NPZ contient les sorties brutes.
Contrôle Monte-Carlo r=25 sur 20000 tirages : erreur moyenne L2=0.015286, erreur covariance relative=0.020916.

### Limites

Une seule gaussienne ne capture pas naturellement les modes correspondant aux différentes classes de chiffres. Les tirages superposent des traits, peuvent sortir de [0,1] et ne sont pas nécessairement réalistes. Plus de variance conservée n'est pas un score perceptuel. Le clipping modifie la loi. Une transformation non linéaire peut représenter des distributions plus complexes ; aucun GAN/VAE n'est entraîné ici.

## Partie 3

### Texte français

Le corpus fourni comporte 166 tokens et 70 états uniques. Le dictionnaire conserve 131 transitions contre 4900 cases pour une matrice dense.
Les dix bigrammes les plus fréquents sont affichés dans le notebook. Le corpus concatène les phrases : point→Une est compté, mais la génération s'arrête sur le point.

Phrase générée demandée : **Une distribution.**
Exemple d'ordre 2 : **Une image représente un chiffre.**
Les phrases d'ordre 1 peuvent manquer de cohérence. Les observations d'ordre 2 ne constituent pas une évaluation linguistique.

### Factorisation

p(x1,x2,x3,x4)=p(x1)p(x2|x1)p(x3|x1,x2)p(x4|x1,x2,x3).
Ordre 1 : conserver seulement le dernier token ; ordre 2 : conserver les deux derniers.
Le premier token est prédit depuis bos, et eos est inclus dans la NLL.

### Lissage et NLL

Laplace α=1 : p(w|c)=(C(c,w)+1)/(somme_v C(c,v)+|V|).
Cela évite les probabilités nulles, au prix d'une diminution de la probabilité des transitions déjà vues. Les vingt séquences de chaque variante figurent dans le notebook et le JSON.

| Modèle | NLL / séquence | NLL / token | Séquences distinctes / 20 |
|---|---:|---:|---:|
| Ordre 1 MLE | 1.994492 | 0.498623 | 8 |
| Ordre 1 Laplace | 4.015042 | 1.003761 | 18 |
| Ordre 2 MLE | 1.732868 | 0.433217 | 6 |
| Ordre 2 Laplace | 4.321576 | 1.080394 | 19 |

Les quatre lots comportent vingt séquences terminées sur eos. Une limite de quinze tokens est définie ; aucun eos n'est ajouté artificiellement pour masquer une troncature.

### Ordre 2 : paramètres et sparsité

Pour une table complète sans conventions de début/fin, V(V−1) paramètres libres à l'ordre 1 et V²(V−1) à l'ordre 2. Dans notre convention explicite : cinq contextes possibles (tous vus) et vingt paramètres libres à l'ordre 1 ; vingt-et-un contextes possibles (huit vus) et quatre-vingt-quatre paramètres libres à l'ordre 2.

L'ordre 2 réduit la NLL d'entraînement sans lissage. Cela ne prouve pas une meilleure généralisation : quatre séquences sont trop peu pour couvrir tous les contextes. Avec Laplace α=1, l'ordre 2 est davantage dilué par les contextes peu observés. Il faut un corpus de validation pour comparer la généralisation. Le nombre de sorties exactement présentes dans le corpus n'est pas un score de qualité absolu.

## Sources

Sujet fourni : [information masquée], Introduction aux modèles génératifs, 20 avril 2026, huit pages.
[PyTorch softplus](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softplus.html), [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).
