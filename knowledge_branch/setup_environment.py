"""
Setup et Configuration de l'environnement pour la branche Knowledge
- Téléchargement des datasets
- Vérification des dépendances
- Configuration des paths
"""

import os
import sys
from pathlib import Path
import torch
import requests
import subprocess

# Définir les chemins de base
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_based"

def print_system_info():
    """Affiche les informations du système disponibles"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    
    print(f'🎮 Device : {device.upper()} - {device_name}')
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'💾 GPU VRAM disponible : {vram:.1f} GB')
    print()

def setup_paths():
    """Configure et valide les chemins du projet"""
    print("📂 Configuration des chemins...")
    print(f"   Répertoire de travail : {KNOWLEDGE_BRANCH}")
    print(f"   ✅ Projet trouvé : {PROJECT_ROOT.exists()}")
    print(f"   ✅ Data trouvée : {DATA_DIR.exists()}")
    
    # Ajouter le knowledge_branch aux imports Python
    if str(KNOWLEDGE_BRANCH) not in sys.path:
        sys.path.insert(0, str(KNOWLEDGE_BRANCH))
    print()

def install_dependencies():
    """Installe les dépendances requises"""
    print("📦 Vérification des dépendances...")
    
    dependencies = [
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('spacy', 'spacy'),
        ('pandas', 'pandas'),
        ('datasets', 'datasets'),
        ('sklearn', 'scikit-learn'),
        ('wikipedia', 'wikipedia-api'),
    ]
    
    missing = []
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - À installer")
            missing.append(package)
    
    if missing:
        print(f"\n📥 Installation de {len(missing)} package(s)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("   ✅ Installation terminée\n")
    else:
        print()

def download_spacy_models():
    """Télécharge les modèles spaCy nécessaires"""
    print("🔤 Téléchargement des modèles spaCy...")
    
    models = ['en_core_web_sm', 'fr_core_news_sm', 'es_core_news_sm']
    
    for model in models:
        try:
            __import__('spacy').load(model)
            print(f"   ✅ {model}")
        except OSError:
            print(f"   ⬇️  Téléchargement de {model}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
    print()

def download_groundtruth_dataset():
    """Télécharge le dataset groundtruth.csv depuis Zenodo"""
    print("📥 Téléchargement du dataset...")
    
    output_file = KNOWLEDGE_BRANCH / "groundtruth.csv"
    
    if output_file.exists():
        print(f"   ✅ Fichier déjà présent : {output_file}")
        return output_file
    
    url = "https://zenodo.org/record/3609356/files/groundtruth.csv?download=1"
    
    try:
        print(f"   ⬇️  Téléchargement depuis Zenodo...")
        response = requests.get(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"   ✅ Fichier téléchargé : {output_file}")
        return output_file
    except Exception as e:
        print(f"   ❌ Erreur lors du téléchargement : {e}")
        return None

def verify_knowledge_modules():
    """Vérifie que les modules principaux de knowledge_branch sont accessibles"""
    print("🔍 Vérification des modules...")
    
    modules = [
        'evidence_retrieval',
        'claim_verification',
        'claim_detection'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ⚠️  {module} - Peut être importé pendant l'exécution")
    print()

def setup_all():
    """Exécute toutes les étapes de setup"""
    print("=" * 50)
    print("🚀 SETUP ENVIRONMENT - KNOWLEDGE BRANCH")
    print("=" * 50)
    print()
    
    print_system_info()
    setup_paths()
    install_dependencies()
    download_spacy_models()
    download_groundtruth_dataset()
    verify_knowledge_modules()
    
    print("=" * 50)
    print("✅ SETUP TERMINÉ")
    print("=" * 50)

if __name__ == "__main__":
    setup_all()
