# Se placer dans le dossier du TP pour retrouver les données et les figures.
Set-Location -LiteralPath $PSScriptRoot

# Réutiliser le Python installé sur ce PC, avec PyTorch et le GPU déjà vérifiés.
$pythonTP2 = 'C:\Users\amjad\AppData\Local\Programs\Python\Python312\python.exe'
if (-not (Test-Path -LiteralPath $pythonTP2)) {
    throw 'Python introuvable : sélectionner un environnement contenant torch et torchvision.'
}

# Ouvrir Jupyter pour lire les commentaires et exécuter les cellules dans l'ordre.
& $pythonTP2 -m jupyter lab '.\TP2_AE_VAE_CVAE_PixelCNN.ipynb'
