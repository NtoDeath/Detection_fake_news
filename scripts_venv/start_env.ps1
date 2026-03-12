#!/usr/bin/env pwsh
# Script de configuration pour le projet Detection_fake_news
# Crée un environnement virtuel Python et installe toutes les dépendances

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Configuration - Projet Détection de Fake News (NLP)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Changer vers le répertoire racine du projet
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptPath
Set-Location $projectRoot

$venvPath = ".venv"

# ========== CRÉATION DE L'ENVIRONNEMENT VIRTUEL ==========
if (Test-Path $venvPath) {
    Write-Host "✅ Environnement virtuel existant détecté" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "📦 Création de l'environnement virtuel..." -ForegroundColor Yellow
    python -m venv $venvPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Environnement virtuel créé!" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "❌ Erreur lors de la création de l'environnement virtuel" -ForegroundColor Red
        Write-Host "   Vérifiez que Python 3.8+ est installé: python --version" -ForegroundColor Yellow
        exit 1
    }
}

# ========== ACTIVATION ==========
Write-Host "🔄 Activation de l'environnement virtuel..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

if ($LASTEXITCODE -eq 0 -or $env:VIRTUAL_ENV) {
    Write-Host "✅ Environnement virtuel activé!" -ForegroundColor Green
    Write-Host ""
    
    # ========== MISE À JOUR DE PIP ==========
    Write-Host "📦 Mise à jour de pip..." -ForegroundColor Yellow
    & python -m pip install --upgrade pip --quiet
    Write-Host "✅ pip à jour" -ForegroundColor Green
    Write-Host ""
    
    # ========== VÉRIFICATION DES DÉPENDANCES ==========
    Write-Host "🔍 Vérification des dépendances principales..." -ForegroundColor Yellow
    
    $packagesInstalled = & python -c @"
import sys
try:
    import pandas, numpy, sklearn, transformers, torch, spacy
    print('ok')
except ImportError:
    print('missing')
"@ 2>$null
    
    if ($packagesInstalled -eq "ok") {
        Write-Host "✅ Dépendances principales déjà installées" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host "📦 Installation des dépendances (cela peut prendre plusieurs minutes)..." -ForegroundColor Yellow
        Write-Host ""
        
        # Installation des packages
        $packages = @(
            "pandas",
            "numpy",
            "scikit-learn",
            "torch",
            "transformers",
            "datasets",
            "evaluate",
            "spacy",
            "nltk",
            "xgboost",
            "textblob",
            "textblob-fr",
            "langdetect",
            "emoji",
            "pyspellchecker",
            "wikipedia-api",
            "requests",
            "matplotlib",
            "seaborn",
            "tqdm",
            "joblib",
            "scipy"
        )
        
        foreach ($package in $packages) {
            Write-Host "  → Installation de $package..." -ForegroundColor Cyan
            & python -m pip install $package --quiet
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    ⚠️ Avertissement: problème avec $package" -ForegroundColor Yellow
            }
        }
        
        Write-Host ""
        Write-Host "✅ Installation des packages terminée!" -ForegroundColor Green
        Write-Host ""
    }
    
    # ========== TÉLÉCHARGEMENT DES MODÈLES SPACY ==========
    Write-Host "🔍 Vérification des modèles spaCy..." -ForegroundColor Yellow
    
    $spacyModelsOk = & python -c @"
try:
    import spacy
    spacy.load('fr_core_news_sm')
    spacy.load('en_core_web_sm')
    print('ok')
except:
    print('missing')
"@ 2>$null
    
    if ($spacyModelsOk -eq "ok") {
        Write-Host "✅ Modèles spaCy déjà installés" -ForegroundColor Green
    } else {
        Write-Host "📦 Téléchargement des modèles spaCy..." -ForegroundColor Yellow
        & python -m spacy download fr_core_news_sm --quiet
        & python -m spacy download en_core_web_sm --quiet
        Write-Host "✅ Modèles spaCy installés!" -ForegroundColor Green
    }
    Write-Host ""
    
    # ========== TÉLÉCHARGEMENT DES DONNÉES NLTK ==========
    Write-Host "🔍 Vérification des données NLTK..." -ForegroundColor Yellow
    
    & python -c @"
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print('✅ NLTK vader_lexicon déjà installé')
except LookupError:
    print('📦 Téléchargement de vader_lexicon...')
    nltk.download('vader_lexicon', quiet=True)
    print('✅ NLTK vader_lexicon installé!')
"@
    
    Write-Host ""
    
    # ========== RÉSUMÉ FINAL ==========
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host " 🎉 Environnement prêt pour la détection de fake news!" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 Environnement Python activé: $venvPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📂 Structure du projet:" -ForegroundColor Yellow
    Write-Host "   • knowledge_branch/  - Détection basée sur les connaissances" -ForegroundColor White
    Write-Host "   • style_branch/      - Détection basée sur le style" -ForegroundColor White
    Write-Host "   • fusion_branch/     - Fusion des méthodes" -ForegroundColor White
    Write-Host "   • data/              - Datasets" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 Pour démarrer:" -ForegroundColor Yellow
    Write-Host "   1. Placez les 2 fichiers ZIP de Teams dans style_branch/" -ForegroundColor White
    Write-Host "   2. Ouvrez et exécutez main.ipynb" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 Commandes utiles:" -ForegroundColor Yellow
    Write-Host "   • jupyter notebook       - Lancer Jupyter" -ForegroundColor White
    Write-Host "   • python main.ipynb      - Exécuter le notebook principal" -ForegroundColor White
    Write-Host "   • deactivate             - Quitter l'environnement virtuel" -ForegroundColor White
    Write-Host ""
    
} else {
    Write-Host "❌ Erreur lors de l'activation de l'environnement virtuel" -ForegroundColor Red
    Write-Host ""
    Write-Host "Essayez d'activer manuellement:" -ForegroundColor Yellow
    Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    exit 1
}
