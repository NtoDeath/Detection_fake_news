# 🚀 Script de Configuration Linux - start_env.sh

Configuration automatisée de l'environnement Python pour le projet **Détection de Fake News** sur Linux.

---

## 📋 Description

Ce script **bash** configure un environnement virtuel Python complet et install toutes les dépendances nécessaires, incluant :
- 📦 **PyTorch avec support GPU CUDA 11.8** (optimisé pour NVIDIA RTX 4060)
- 🔄 **Packages ML/NLP** (transformers, scikit-learn, spacy, xgboost, etc.)
- 📓 **Jupyter & IPyKernel** pour les notebooks
- 🔌 **Kernel Jupyter enregistré** automatiquement dans VS Code

---

## ✅ Prérequis

### Obligatoire
- **Python 3.8+** installé sur le système
- **Bash shell** (natif sur Linux)
- Connexion Internet

### Recommandé (pour accélération GPU)
- **GPU NVIDIA** (testé sur GTX 4060)
- **NVIDIA drivers 550+** installés
- **CUDA 11.8+** (le script gère automatiquement)

---

## 🎯 Utilisation

### Depuis la racine du projet

```bash
./scripts_venv/start_env.sh
```

### Depuis le dossier scripts_venv

```bash
cd scripts_venv && ./start_env.sh
```

**Première exécution :** ⏱️ 5-15 minutes (téléchargement packages)  
**Exécutions suivantes :** ⚡ 30-60 secondes (vérification uniquement)

---

## 🔍 Ce que le script fait

### 1. Création de l'environnement virtuel
```bash
python3 -m venv .venv
```
Crée un dossier `.venv/` isolé (~3-5 GB).

### 2. Activation automatique
```bash
source .venv/bin/activate
```

### 3. Installation de PyTorch avec GPU
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Installe **PyTorch 2.7+ avec CUDA 11.8** si GPU détecté
- Fallback vers PyTorch CPU si GPU non disponible
- Compatible avec RTX 4060, RTX 3060, RTX 4090, etc.

### 4. Installation des dépendances
Packages installés automatiquement :
- **Data:** pandas, numpy, scipy
- **ML:** scikit-learn, xgboost
- **NLP:** transformers, datasets, evaluate, spacy, nltk
- **Utils:** matplotlib, seaborn, tqdm, joblib, emoji, requests
- **NLP avancé:** textblob, textblob-fr, langdetect, pyspellchecker, wikipedia-api
- **Notebooks:** jupyter, ipykernel

### 5. Téléchargement des modèles linguistiques
- **spaCy:** `fr_core_news_sm` (français) + `en_core_web_sm` (anglais)
- **NLTK:** `vader_lexicon` (analyse sentiment)

### 6. Création du kernel Jupyter
```bash
python -m ipykernel install --user --name fake-news-venv \
  --display-name "Python (Fake News venv)"
```
Le kernel est automatiquement disponible dans VS Code.

### 7. Vérification GPU
Affiche le statut CUDA et le device GPU détecté.

---

## 💻 Statut GPU

Le script affiche automatiquement:

```
💻 Statut GPU:
CUDA: True
Device: NVIDIA GeForce RTX 4060
```

**Si CPU only:**
```
CUDA: False
Device: CPU only
```

---

## 🔧 Configuration avancée

### Désactiver l'installation GPU
Si vous n'avez pas de GPU ou préférez CPU, modifiez la ligne:

```bash
# Avant (avec GPU):
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet

# Après (CPU only):
python -m pip install torch torchvision torchaudio --quiet
```

### Ajouter des packages supplémentaires
Modifiez le tableau `PACKAGES`:

```bash
PACKAGES=(
    "pandas"
    "numpy"
    # ... packages existants ...
    "votre-nouveau-package"  # Ajoutez ici
)
```

### Modifier le timeout d'installation
Augmentez le timeout pip pour les connexions lentes:

```bash
pip install $package --quiet --default-timeout=1000 2>/dev/null
```

---

## 🔌 Utilisation du kernel dans VS Code

### Option 1 - Sélection manuelle
1. Ouvrez `main.ipynb` dans VS Code
2. Cliquez sur **"Sélectionner le kernel"** (haut droit)
3. Choisissez **"Python (Fake News venv)"**

### Option 2 - Configuration automatique
Créez ou modifiez `.vscode/settings.json` à la racine du projet:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "notebook.kernelShutdownOnLastWindowClose": true
}
```

---

## 🛠️ Adaptation à d'autres notebooks

### Pour un nouveau projet

1. **Dupliquez le script** :
   ```bash
   cp scripts_venv/start_env.sh scripts_venv/start_env_mon_projet.sh
   ```

2. **Modifiez les noms du kernel** :
   ```bash
   # Avant:
   --name fake-news-venv --display-name "Python (Fake News venv)"
   
   # Après:
   --name mon-projet-venv --display-name "Python (Mon Projet)"
   ```

3. **Adaptez la liste des packages** :
   ```bash
   PACKAGES=(
       "pandas"
       "numpy"
       # Packages spécifiques à votre projet
       "requests"
       "beautifulsoup4"
       "selenium"
   )
   ```

4. **Modifiez les messages finaux** :
   ```bash
   echo " 🎉 Environnement prêt pour MON PROJET!"
   ```

### Exemple: Adaptation pour projet Computer Vision

```bash
# Remplacez la liste PACKAGES par:
PACKAGES=(
    "opencv-python"
    "pillow"
    "scikit-image"
    "torch"  # ⚠️ PyTorch sera installé séparément
    "torchvision"
    "matplotlib"
    "numpy"
    "pandas"
    "jupyter"
    "ipykernel"
)

# Et le kernel Jupyter:
--name cv-project-venv --display-name "Python (Computer Vision)"
```

### Créer un script générique

Créez `scripts_venv/start_env_template.sh` pour tous les projets:

```bash
#!/bin/bash
PROJECT_NAME="${1:-mon-projet}"
KERNEL_NAME="${2:-${PROJECT_NAME}-venv}"
DISPLAY_NAME="${3:-Python (${PROJECT_NAME})}"

# ... reste du script ...

python -m ipykernel install --user --name "$KERNEL_NAME" \
  --display-name "$DISPLAY_NAME" 2>/dev/null
```

Utilisation:
```bash
./scripts_venv/start_env_template.sh "fake-news" "fake-news-venv" "Python (Fake News venv)"
```

---

## 🐛 Dépannage

### Le script ne s'exécute pas
```bash
chmod +x ./scripts_venv/start_env.sh
./scripts_venv/start_env.sh
```

### Le kernel n'apparaît pas dans VS Code
1. **Redémarrez** VS Code
2. Exécutez manuellement:
   ```bash
   source .venv/bin/activate
   python -m ipykernel install --user --name fake-news-venv \
     --display-name "Python (Fake News venv)" --force
   ```

### Installation GPU échoue
Le script **fallback automatiquement** vers CPU. Vérifiez:
```bash
nvidia-smi  # Doit afficher votre GPU
nvcc --version  # CUDA toolkit
```

### Packages non installés
Relancez le script, certains téléchargements peuvent être lents:
```bash
rm -rf .venv
./scripts_venv/start_env.sh
```

### Venv très volumineux
Le dossier `.venv/` peut atteindre **3-5 GB**. C'est normal.
Supprimez les packages inutilisés:
```bash
source .venv/bin/activate
pip cache purge
```

---

## 📊 Performance GPU

Avec **RTX 4060 + CUDA 11.8**:

| Tâche | CPU | GPU | Speedup |
|-------|-----|-----|---------|
| Fine-tuning RoBERTa | 2-4h | 30-45 min | **3-5x** |
| Feature extraction | 10-15 min | 2-5 min | **2-3x** |
| Inference | Sequential | Batch | **2-10x** |

---

## 📝 Commandes utiles après activation

```bash
# Activer le venv manuel
source .venv/bin/activate

# Afficher les packages installés
pip list

# Vérifier GPU
python -c "import torch; print(torch.cuda.is_available())"

# Lancer Jupyter
jupyter notebook

# Quitter le venv
deactivate
```

---

## 📄 Fichiers créés

- `.venv/` — Environnement virtuel Python
- `~/.var/app/com.visualstudio.code/data/jupyter/kernels/fake-news-venv/` — Configuration kernel VS Code

---

## 📞 Support

Pour modifier ce script pour d'autres projets, consultez la section **"Adaptation à d'autres notebooks"** ci-dessus.

Dernière mise à jour: **Avril 2026**
