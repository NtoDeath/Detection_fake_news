# Knowledge Branch - Scripts de Fact-Checking

Ce répertoire contient les scripts Python modulaires pour le pipeline de vérification de fausses nouvelles basé sur la connaissance (Knowledge-based Fake News Detection).

## 📁 Structure des fichiers

```
knowledge_branch/
├── knowledge_main.ipynb           # Notebook orchestrateur principal
├── setup_environment.py           # 1. Configuration initiale
├── train_claim_detector.py        # 2. Entraînement DistilBERT
├── initialize_pipeline.py         # 3. Initialisation des composants
├── evaluate_pipeline.py           # 4. Évaluation sur dataset FEVER
├── full_pipeline.py               # 5. Pipeline complet
│
├── evidence_retrieval.py          # Module: récupération de preuves (existant)
├── claim_verification.py          # Module: vérification (existant)
├── claim_detection.py             # Module: détection de claims (existant)
│
├── results/                       # Sorties
│   ├── confusion_matrix.png       # Visualisation
│   └── evaluation_report.txt      # Rapport d'évaluation
│
└── claim_detector_model/          # Modèle fine-tuné (après phase 2)
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## 🚀 Exécution rapide

### Option 1 : Notebook (Recommandé)
```bash
cd knowledge_branch
jupyter notebook knowledge_main.ipynb
```
puis exécutez les cellules dans l'ordre.

### Option 2 : Scripts individuels
```bash
cd knowledge_branch

# Phase 1: Setup
python setup_environment.py

# Phase 2: Entraînement claim detector
python train_claim_detector.py

# Phase 3: Initialisation
python initialize_pipeline.py

# Phase 4: Évaluation
python evaluate_pipeline.py

# Phase 5: Pipeline complet
python full_pipeline.py
```

## 📋 Description des phases

### Phase 1 : `setup_environment.py`
**Durée:** 5-10 minutes

Prépare l'environnement:
- Installe les dépendances manquantes
- Télécharge les modèles spaCy (EN, FR, ES)
- Télécharge le dataset groundtruth.csv depuis Zenodo
- Vérifie que tous les modules sont accessibles

**Sortie:** 
- `groundtruth.csv` (5000+ samples)
- Modèles spaCy installés

### Phase 2 : `train_claim_detector.py`
**Durée:** 5-15 min (GPU), 20-40 min (CPU)

Fine-tune DistilBERT pour la détection de claims:
- Charge groundtruth.csv
- Split 80/20 (train/test)
- Fine-tune 3 epochs, batch=16, lr=2e-5
- Évalue avec matrice de confusion et F1-score
- Sauvegarde le modèle dans `my_claim_model/`

**Entrée:** `groundtruth.csv`
**Sortie:** `my_claim_model/` (config.json, model.safetensors, tokenizer.json)

**Hyperparamètres:**
- Modèle: distilbert-base-uncased
- Learning rate: 2e-5
- Epochs: 3
- Batch size: 16
- Max tokens: 512

### Phase 3 : `initialize_pipeline.py`
**Durée:** 2-3 minutes

Initialise et teste les composants principaux:
- Lance EvidenceRetriever avec APIs configurées
- Lance ClaimVerifier avec modèle RoBERTa NLI
- Teste sur exemples multi-langues (EN, FR, ES)
- Affiche des résultats pour validation

**Configuration API:**
- Wolfram Alpha: `LEU7Y6728T`
- Google Custom Search Engine: `151bf4aa4eae44373`
- Google API Key: À configurer (optionnel)

**Tests inclus:**
- "The Eiffel Tower is 330 meters tall" (EN)
- "Le taux de chômage en France est de 7%" (FR)
- "La capital de España es Madrid" (ES)

### Phase 4 : `evaluate_pipeline.py`
**Durée:** 10-20 minutes

Évalue complètement le pipeline sur dataset FEVER:
- Charge `data/knowledge_based/train.jsonl`
- Sélectionne 30 samples par classe (SUPPORTED, REFUTED, NEUTRAL)
- Lance retriever et verifier sur tous les claims
- Génère rapports et visualisations

**Métriques:**
- Precision, Recall, F1-score par classe
- Accuracy globale
- Taux de récupération par classe
- Matrice de confusion

**Sorties:**
- `results/evaluation_report.txt`
- `results/confusion_matrix.png`

### Phase 5 : `full_pipeline.py`
**Durée:** Variable (selon APIs externes)

Pipeline complet pour traiter du texte arbitraire:

1. **Entrée:** Texte en EN/FR/ES
2. **Extraction:** Phrases avec spaCy
3. **Détection:** Entités nommées (GPE, PERSON, DATE, ORG) + ML claim scoring
4. **Récupération:** Evidence retriever avec priorité des sources
5. **Vérification:** NLI score pour chaque claim
6. **Sortie:** Rapport JSON avec verdicts

**Verdict possible:** SUPPORTED, REFUTED, NEUTRAL / NOT ENOUGH INFO, NO_EVIDENCE, NOT_CHECKED

**Fonctions principales:**
- `main(text, language)` - Traiter un texte
- `test_multiple_languages()` - Test EN/FR/ES

**Exemple d'utilisation en code:**
```python
from full_pipeline import main

text = "Paris is the capital of France. The Earth is flat."
report = main(text, language='en')
```

## 📊 Architecture du pipeline

```
Texte
  ↓
[Initialisation]
├── EvidenceRetriever (spaCy + Google + Wolfram)
├── ClaimVerifier (RoBERTa NLI)
└── ClaimDetector (DistilBERT, optionnel)
  ↓
[Traitement par phrase]
  ├── NER: Extraction entités
  ├── ML: Score claim (optionnel)
  ├── Décision: Vérifier?
  ├── Retrievel: Chercher preuves
  └── Verify: NLI scoring
  ↓
[Rapport JSON]
├── phrase
├── verdict (SUPPORTED|REFUTED|NEUTRAL)
├── confidence
├── source
├── entities
└── evidence_snippet
```

## 🔧 Configuration

### Localisation des fichiers
```python
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_based"
```

### Clés API à configurer
Dans `initialize_pipeline.py` et `full_pipeline.py`:
```python
GOOGLE_API_KEY = None  # À remplacer
GOOGLE_CSE_ID = "151bf4aa4eae44373"
WOLFRAM_APPID = "LEU7Y6728T"
```

## 📦 Dépendances

Installées automatiquement par `setup_environment.py`:
- transformers
- torch
- spacy (+ modèles en français et espagnol)
- pandas
- datasets
- scikit-learn
- wikipedia-api
- matplotlib
- seaborn

## 🎯 Cas d'utilisation

### Vérifier un article complet
```bash
python full_pipeline.py
```

### Évaluer les performances
```bash
python evaluate_pipeline.py
```

### Réentraîner le claim detector
```bash
python train_claim_detector.py
```

### Tester multi-langues en Python
```python
from full_pipeline import main
main("Le COVID-19 est originaire de Wuhan", language='fr')
```

## 📈 Résultats attendus

**Claim Detector (Phase 2):**
- Accuracy: >95%
- Precision: >94%
- Recall: >94%
- F1-score: >94%

**Pipeline complet (Phase 4):**
- Taux de récupération: 60-80% (dépend de qualité preuves)
- Accuracy NLI: 65-75%
- F1-score par classe: 60-75%

## ⚠️ Limitations

1. **Dataset FEVER limité** : 90 samples pour évaluation (30 par classe)
2. **APIs externes** : Google Search et Wolfram Alpha limités
3. **Langues supportées** : EN, FR, ES uniquement
4. **Performance NLI** : RoBERTa ne garantit pas 100% accuracy
5. **Preuves incomplètes** : Pas toujours possible de trouver des preuves fiables

## 🔄 Workflow recommandé

```
1. setup_environment.py          → Préparation
   ↓
2. train_claim_detector.py       → 15 min (GPU) ou 40 min (CPU)
   ↓
3. initialize_pipeline.py        → Validation
   ↓
4. evaluate_pipeline.py          → Test quantitatif
   ↓
5. full_pipeline.py              → Production
```

## 📝 Notes

- Les scripts utilisent `sys.path` pour résoudre les imports locaux
- Les modèles sont téléchargés une première fois (peut être long)
- Les résultats d'extraction sur APIs externes dépendent de leur disponibilité
- Recommandé d'utiliser une GPU pour l'entraînement (Phase 2)

## 🤝 Support

Pour des questions sur l'utilisation, consultez:
- `knowledge_main.ipynb` pour vue d'ensemble
- Docstrings au début de chaque script (fonction `main()`)
- Code source des modules: `evidence_retrieval.py`, `claim_verification.py`
