# IA générative visuelle

Collection de notebooks, expériences et comptes rendus sur les modèles génératifs.

## Organisation et état

| Travail | Contenu | État des livrables |
|---|---|---|
| [TP1](TP1_Modeles_Generatifs/) | Probabilités, génération d'images latentes, modèles autorégressifs | Trois notebooks avec sorties enregistrées, figures, mesures et rapport |
| [TP2](TP2_AE_VAE_CVAE_PixelCNN/) | AE, VAE, CVAE et PixelCNN sur MNIST | Notebook avec sorties enregistrées, figures et rapport |
| [TP3 — GAN](TP3_GAN/TP3_GAN.ipynb) | Cercle, demi-lunes, Swiss roll, cGAN, DCGAN, LeNet5, SVM et espace latent | Exécution et résultats documentés dans le notebook |
| [Mini-projet — StyleGAN](Mini_Projet_StyleGAN/Mini_Projet_StyleGAN.ipynb) | Mapping, AdaIN, bruit, mélange de styles et exploration de W | Exécution et résultats documentés dans le notebook |
| [Adaptation CIFAR-10](Adaptation_CIFAR10_incomplete/) | Variante RGB des modèles du TP2 | **Incomplète** : sorties partielles de l'AE ; VAE, CVAE et PixelCNN non exécutés |

Les notebooks contiennent directement leurs sorties : ils peuvent être consultés sur GitHub sans lancer d'entraînement. Les fichiers `resultats.json` et les figures accompagnent les nouveaux travaux. Les sorties historiques sont conservées ; elles ne sont pas présentées comme une nouvelle exécution.

## Poids de remplacement et limites

Les poids DCGAN et LeNet5 mentionnés dans le sujet GAN n'étaient pas fournis. Les deux modèles sont **entraînés localement en remplacement**, sur MNIST. Aucune équivalence avec les modèles du professeur n'est revendiquée. LeNet5 génère des pseudo-étiquettes pour les images synthétiques ; une concordance avec ces étiquettes ne mesure pas une vérité indépendante.

Le premier essai Mini-StyleGAN effondré est conservé ; le second ajoute une statistique de diversité au discriminateur. Chacun utilise 15 époques. Les expériences GAN et le second Mini-StyleGAN utilisent 15 époques sur 60 000 images MNIST ; LeNet5 utilise 3 époques. Les expériences SVM utilisent 10 000 images réelles d'apprentissage et 1 000 synthétiques. Les détails et observations figurent dans les notebooks. Les grilles montrent des échantillons réels du calcul, sans retouche ni sélection manuelle d'images favorables.

Ce Mini-StyleGAN pédagogique à résolution 28×28 n'est pas une reproduction de StyleGAN2/3 haute résolution. Les graines sont fixées, mais les versions, le matériel et les calculs GPU peuvent modifier les résultats d'une nouvelle exécution.

## Installation et exécution — PowerShell

```powershell
git clone https://github.com/AmjadAAYD/IA-Generative-Visuelle.git
Set-Location -LiteralPath '.\IA-Generative-Visuelle'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name ia-generative-visuelle --display-name 'IA générative visuelle'
.\.venv\Scripts\python.exe -m jupyterlab
```

Ouvrir le notebook du TP souhaité, sélectionner le noyau de l'environnement créé puis exécuter les cellules dans l'ordre. Pour les deux nouveaux notebooks, garder le répertoire du notebook comme répertoire courant : MNIST est téléchargé dans `../data/`. Un GPU CUDA compatible accélère fortement les entraînements ; le code sait aussi utiliser MPS ou le CPU. Une installation PyTorch adaptée à son accélérateur peut être nécessaire ; `requirements.txt` ne fige pas une version CUDA propre à une seule machine.

Les jeux téléchargés, environnements et checkpoints `.pt` sont exclus de Git. Une nouvelle exécution recrée les modèles dans le sous-dossier `models/` de chaque TP. Les petits fichiers NPZ du TP1 sont inclus pour conserver ses données et résultats. Les sujets des enseignants ne sont pas redistribués.

## Lecture des résultats

Les pertes adversariales ne suffisent pas à établir la qualité ou la diversité des générations. Les figures, la distribution des pseudo-classes et le transfert SVM sont présentés ensemble. Les tests de dimensions et de valeurs finies vérifient l'exécution ; ils ne prouvent pas la qualité perceptuelle. Aucun FID ou score de généralisation non calculé n'est annoncé.
