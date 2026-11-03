# Introduction aux modèles génératifs

**Binôme : Amjad AAYD et [information masquée] — [information masquée] — 6 octobre 2026**
Enseignant : [information masquée]. Groupe : à compléter.
Sujet : document fourni, huit pages, 20 avril 2026. Les trois parties sont réalisées.

## Livrables

- 01_Modeles_probabilistes.ipynb : Bernoulli, gaussiennes 1D/3D, EM GMM 3D manuel et GMM sinusoïdal 2D.
- 02_Generation_images_latentes.ipynb : digits, covariance /N, eigen-images, génération r=10/25/50 et directions latentes.
- 03_Modeles_autoregressifs.ipynb : corpus français, bigrammes, ordre 2, Laplace, NLL et quatre lots de vingt séquences.
- Compte_Rendu_Modeles_Generatifs.pdf : rapport détaillé avec les quinze figures et résultats.
- Reponses_et_Resultats.md : réponses théoriques et résultats essentiels.
- figures/ : quinze figures PNG originales.
- metriques_partie1/2/3.json : résultats non arrondis.
- *_sorties.txt : sorties textuelles des cellules exécutées.
- corpus_francais.txt, sequences_generees.json : données textuelles et quatre lots générés.
- donnees_partie1.npz, modele_images.npz : données synthétiques et paramètres/sorties brutes du modèle d'images.
- versions_environnement.json : instantané exact des bibliothèques utilisées.

## Relancer sur ce poste

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\Lancer_Notebooks.ps1
```

Le script utilise l'environnement de travail distinct, qui hérite des bibliothèques scientifiques Python 3.12 déjà installées avec system-site-packages. Choisir le kernel **Python 3.12 (TP modèles génératifs)**. Exécuter les cellules de chaque notebook dans l'ordre. Les calculs utilisent le CPU, float64 et deux threads ; aucun GPU n'est nécessaire.

Les versions exactes, dont la version de développement PyTorch préexistante, sont indiquées dans le JSON. Les contraintes ci-dessous servent à une installation neuve ; elles ne constituent pas une installation neuve vérifiée sur un autre poste, et les derniers chiffres peuvent dépendre des versions.

## Installation portable

Depuis ce dossier sur un poste avec Python compatible :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_portables.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tp-generatifs --display-name "Python 3.12 (TP modèles génératifs)"
.\.venv\Scripts\python.exe -m jupyterlab
```

## Conventions importantes

- Échantillon Bernoulli : N=1000, p vrai=0.7, MLE réalisé=0.696.
- NLL : log naturel, moyenne par observation ; pour les séquences, résultats par séquence et par token (fin incluse).
- Covariance MLE/empirique : diviseur N, conformément au sujet.
- Bruit sinus : variance 0.1, donc écart-type sqrt(0.1).
- Images : 1500 images digits locales, 8x8. Sorties gaussiennes brutes conservées ; clipping seulement pour l'affichage.
- Générations symboliques : arrêt sur eos ou limite de quinze tokens, sans ajout forcé de eos. Les quatre lots observés sont tous terminés.
- L'estimation reproduit le MLE de l'échantillon, pas exactement les valeurs vraies. Les statistiques de génération ne prouvent pas une qualité perceptuelle ou linguistique générale.
