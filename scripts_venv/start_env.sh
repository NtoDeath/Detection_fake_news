#!/bin/bash

# Script de configuration pour le projet Detection_fake_news (Linux)
# Crée un environnement virtuel Python et installe toutes les dépendances

echo ""
echo "========================================================"
echo " Configuration - Projet Détection de Fake News (NLP)"
echo "========================================================"
echo ""

# Obtenir le chemin du répertoire racine du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

VENV_PATH=".venv"

# ========== CRÉATION DE L'ENVIRONNEMENT VIRTUEL ==========
if [ -d "$VENV_PATH" ]; then
    echo "✅ Environnement virtuel existant détecté"
    echo ""
else
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv "$VENV_PATH"
    
    if [ $? -eq 0 ]; then
        echo "✅ Environnement virtuel créé!"
        echo ""
    else
        echo "❌ Erreur lors de la création de l'environnement virtuel"
        echo "   Vérifiez que Python 3.8+ est installé: python3 --version"
        exit 1
    fi
fi

# ========== ACTIVATION ==========
echo "🔄 Activation de l'environnement virtuel..."
source "$VENV_PATH/bin/activate"

if [ $? -eq 0 ] && [ ! -z "$VIRTUAL_ENV" ]; then
    echo "✅ Environnement virtuel activé!"
    echo ""
    
    # ========== MISE À JOUR DE PIP ==========
    echo "📦 Mise à jour de pip..."
    python -m pip install --upgrade pip --quiet 2>/dev/null
    echo "✅ pip à jour"
    echo ""
    
    # ========== VÉRIFICATION DES DÉPENDANCES ==========
    echo "🔍 Vérification des dépendances principales..."
    
    PACKAGES_INSTALLED=$(python -c "
try:
    import pandas, numpy, sklearn, transformers, torch, spacy
    print('ok')
except ImportError:
    print('missing')
" 2>/dev/null)
    
    if [ "$PACKAGES_INSTALLED" = "ok" ]; then
        echo "✅ Dépendances principales déjà installées"
        echo ""
    else
        echo "📦 Installation des dépendances (cela peut prendre plusieurs minutes)..."
        echo ""
        
        # Installation de PyTorch avec support CUDA (pour GPU NVIDIA)
        echo "  → Installation de PyTorch avec CUDA 11.8..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "    ✅ PyTorch (GPU) installé!"
        else
            echo "    ⚠️ Fallback: installation de PyTorch CPU..."
            python -m pip install torch torchvision torchaudio --quiet 2>/dev/null
        fi
        
        # Installation des packages supplémentaires
        PACKAGES=(
            "pandas"
            "numpy"
            "scikit-learn"
            "transformers"
            "datasets"
            "evaluate"
            "accelerate"
            "bitsandbytes"
            "spacy"
            "nltk"
            "xgboost"
            "textblob"
            "textblob-fr"
            "langdetect"
            "emoji"
            "pyspellchecker"
            "wikipedia-api"
            "requests"
            "matplotlib"
            "seaborn"
            "tqdm"
            "joblib"
            "scipy"
            "jupyter"
            "ipykernel"
        )
        
        for package in "${PACKAGES[@]}"; do
            echo "  → Installation de $package..."
            python -m pip install "$package" --quiet 2>/dev/null
            if [ $? -ne 0 ]; then
                echo "    ⚠️ Avertissement: problème avec $package"
            fi
        done
        
        echo ""
        echo "✅ Installation des packages terminée!"
        echo ""
    fi
    
    # ========== TÉLÉCHARGEMENT DES MODÈLES SPACY ==========
    echo "🔍 Vérification des modèles spaCy..."
    
    SPACY_MODELS_OK=$(python -c "
try:
    import spacy
    spacy.load('fr_core_news_sm')
    spacy.load('en_core_web_sm')
    print('ok')
except:
    print('missing')
" 2>/dev/null)
    
    if [ "$SPACY_MODELS_OK" = "ok" ]; then
        echo "✅ Modèles spaCy déjà installés"
    else
        echo "📦 Téléchargement des modèles spaCy..."
        python -m spacy download fr_core_news_sm --quiet 2>/dev/null
        python -m spacy download en_core_web_sm --quiet 2>/dev/null
        echo "✅ Modèles spaCy installés!"
    fi
    echo ""
    
    # ========== TÉLÉCHARGEMENT DES DONNÉES NLTK ==========
    echo "🔍 Vérification des données NLTK..."
    
    python -c "
import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print('✅ NLTK vader_lexicon déjà installé')
except LookupError:
    print('📦 Téléchargement de vader_lexicon...')
    nltk.download('vader_lexicon', quiet=True)
    print('✅ NLTK vader_lexicon installé!')
" 2>/dev/null
    
    echo ""
    
    # ========== CRÉATION DU KERNEL IPYTHON ==========
    echo "🔌 Création du kernel Jupyter pour VS Code..."
    python -m ipykernel install --user --name fake-news-venv --display-name "Python (Fake News venv)" 2>/dev/null
    echo "✅ Kernel Jupyter enregistré!"
    echo ""
    
    # ========== RÉSUMÉ FINAL ==========
    echo "========================================================"
    echo " 🎉 Environnement prêt pour la détection de fake news!"
    echo "========================================================"
    echo ""
    echo "📍 Environnement Python activé: $VENV_PATH"
    echo ""
    echo "📂 Structure du projet:"
    echo "   • knowledge_branch/  - Détection basée sur les connaissances"
    echo "   • style_branch/      - Détection basée sur le style"
    echo "   • fusion_branch/     - Fusion des méthodes"
    echo "   • data/              - Datasets"
    echo ""
    echo "🚀 Pour démarrer:"
    echo "   1. Placez les 2 fichiers ZIP de Teams dans style_branch/"
    echo "   2. Ouvrez et exécutez main.ipynb"
    echo ""
    echo "💡 Commandes utiles:"
    echo "   • jupyter notebook       - Lancer Jupyter en local"
    echo "   • code main.ipynb        - Ouvrir le notebook dans VS Code"
    echo "   • deactivate             - Quitter l'environnement virtuel"
    echo ""
    echo "📓 Pour VS Code:"
    echo "   1. Ouvrez main.ipynb"
    echo "   2. Cliquez sur 'Sélectionner le kernel' (en haut à droite)"
    echo "   3. Choisissez 'Python (Fake News venv)'"
    echo ""
    
    # ========== VÉRIFICATION GPU ==========
    echo "💻 Statut GPU:"
    CUDA_CHECK=$(python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>/dev/null)
    echo "$CUDA_CHECK"
    echo ""
    
else
    echo "❌ Erreur lors de l'activation de l'environnement virtuel"
    echo ""
    echo "Essayez d'activer manuellement:"
    echo "   source .venv/bin/activate"
    exit 1
fi
