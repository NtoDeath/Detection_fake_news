# 🚀 Démarrage de l'environnement virtuel Python

## 📋 Description

Ce script PowerShell configure automatiquement l'environnement de développement pour le projet de détection de fake news. Il crée un environnement virtuel Python isolé et installe toutes les dépendances nécessaires.

## ✅ Prérequis

- **Python 3.8+** installé sur votre système
- **PowerShell** (inclus dans Windows)
- Connexion Internet pour télécharger les packages

## 🎯 Utilisation

### Depuis la racine du projet

```powershell
.\scripts_venv\start_env.ps1
```

### Depuis le dossier scripts_venv

```powershell
.\start_env.ps1
```

## 📦 Ce qui est installé

### Packages Python principaux
- **pandas**, **numpy** - Manipulation de données
- **scikit-learn** - Machine Learning
- **torch** - Deep Learning (PyTorch)
- **transformers**, **datasets**, **evaluate** - NLP avec Hugging Face
- **spacy**, **nltk** - Traitement du langage naturel
- **xgboost** - Algorithmes de boosting
- **textblob**, **textblob-fr** - Analyse de texte FR/EN
- **langdetect** - Détection de langue
- **emoji** - Gestion des emojis
- **pyspellchecker** - Vérification orthographique
- **wikipedia-api** - Accès à Wikipedia
- **matplotlib**, **seaborn** - Visualisation
- **tqdm**, **joblib**, **scipy** - Utilitaires

### Modèles linguistiques
- **spaCy**: `fr_core_news_sm` (français) et `en_core_web_sm` (anglais)
- **NLTK**: `vader_lexicon` (analyse de sentiment)

## ⚙️ Fonctionnement

1. **Création du venv** : Crée `.venv` à la racine du projet (si absent)
2. **Activation** : Active automatiquement l'environnement virtuel
3. **Installation** : Installe tous les packages requis
4. **Téléchargement** : Récupère les modèles spaCy et données NLTK
5. **Confirmation** : Affiche un résumé de la configuration

## 🔧 Commandes utiles après activation

```powershell
# Lancer Jupyter Notebook
jupyter notebook

# Vérifier les packages installés
pip list

# Quitter l'environnement virtuel
deactivate
```

## 🐛 Dépannage

### Erreur d'exécution de script PowerShell

Si vous obtenez une erreur de politique d'exécution :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### L'environnement ne s'active pas

Activez manuellement :

```powershell
.\.venv\Scripts\Activate.ps1
```

### Installation échoue

Vérifiez votre version de Python :

```powershell
python --version
```

Mettez à jour pip manuellement :

```powershell
python -m pip install --upgrade pip
```

## 📝 Notes importantes

- ⏱️ La **première exécution** peut prendre 5-10 minutes (téléchargement des packages)
- 📁 Les exécutions suivantes sont **quasi instantanées** (vérification uniquement)
- 💾 L'environnement `.venv` prend environ **2-3 Go** d'espace disque
- ⚡ Tous les packages sont installés **localement** dans `.venv` (pas de pollution globale)

## 🎓 Projet

Ce script est configuré pour le projet **IFT714 - Détection de Fake News** qui utilise :
- Détection basée sur les **connaissances** (knowledge_branch)
- Détection basée sur le **style** (style_branch)
- **Fusion** des deux approches (fusion_branch)
