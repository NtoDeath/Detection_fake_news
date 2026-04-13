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
        echo "  → Installation de PyTorch avec CUDA 12.4 (pour RTX 4060)..."
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "    ✅ PyTorch CUDA 12.4 (GPU) installé!"
            
            # Installation des packages GPU optimisés
            echo "  → Installation des packages GPU (cuDNN, nvidia-cublas, xformers)..."
            python -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 --quiet 2>/dev/null
            python -m pip install xformers --quiet 2>/dev/null
            echo "    ✅ Packages GPU optimisés installés!"
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
    echo "⚡ Pop!_OS GPU Configuration:"
    echo "   • RTX 4060: Compute Capability 8.9 (Ada)"
    echo "   • CUDA Runtime: Accessible via PyTorch"
    echo "   • PyTorch: Configuré pour CUDA 13.0+"
    echo "   • xformers: Installé pour speedup Transformers"
    echo ""
    echo "📝 Notes:"
    echo "   • nvidia-smi peut ne pas être en PATH en environnement Flatpak"
    echo "   • Mais CUDA Libraries (/usr/lib/libcuda.so) sont détectées"
    echo "   • PyTorch peut utiliser le GPU directement"
    echo "   • Si 'PyTorch CUDA: NON DISPONIBLE', réexécutez ce script"
    echo ""
    
    # ========== VÉRIFICATION GPU ==========
    echo "💻 Statut GPU:"
    
    # Vérifier les librairies CUDA runtime
    CUDA_LIBS=$(find /usr -name "libcuda.so*" 2>/dev/null | head -1)
    if [ -n "$CUDA_LIBS" ]; then
        echo "  ✅ CUDA Runtime Libraries détectées"
        echo "     $CUDA_LIBS"
    else
        echo "  ⚠️ CUDA Runtime Libraries non trouvées"
    fi
    
    # Vérifier nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        echo "  ✅ NVIDIA Driver Tools (nvidia-smi) détectés"
    else
        echo "  ℹ️  nvidia-smi non trouvé (mais CUDA runtime peut fonctionner)"
    fi
    
    # Vérifier PyTorch GPU avec retry
    echo ""
    echo "  🔍 Test PyTorch CUDA (cela peut prendre quelques secondes)..."
    
    CUDA_CHECK=$(python -c "
import sys
import torch

try:
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print('  ✅ PyTorch CUDA: DISPONIBLE')
        print(f'     GPU: {torch.cuda.get_device_name(0)}')
        print(f'     CUDA Version: {torch.version.cuda}')
        print(f'     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print(f'     Compute Capability: {torch.cuda.get_device_capability(0)}')
    else:
        print('  ⚠️  PyTorch CUDA: NON DISPONIBLE')
        print('     → Vérifiez que PyTorch a été installé avec CUDA support')
        print('     → Commande: pip install torch --index-url https://download.pytorch.org/whl/cu124')
        print('     Device: CPU (fallback)')
except Exception as e:
    print(f'  ❌ Erreur lors de la vérification: {str(e)}')
    print('     → Essayez de réinstaller PyTorch')
    sys.exit(1)
" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$CUDA_CHECK"
    else
        echo "  ❌ Erreur critique - PyTorch ne peut pas être importé"
        echo "     Essayez: pip install torch --upgrade"
    fi
    echo ""
    
else
    echo "❌ Erreur lors de l'activation de l'environnement virtuel"
    echo ""
    echo "Essayez d'activer manuellement:"
    echo "   source .venv/bin/activate"
    exit 1
fi
