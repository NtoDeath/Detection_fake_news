# Comparaison: Part A (80%) vs Dataset Complet (100%)

**Date:** 10 avril 2026  
**Auteur:** Pipeline de Détection de Fausses Nouvelles  
**Objectif:** Analyser l'impact de la réduction de 20% des données d'entraînement sur la performance des modèles

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Méthodologie](#méthodologie)
3. [Résultats RoBERTa Baseline](#résultats-roberta-baseline)
4. [Résultats Ensemble (RandomForest + XGBoost)](#résultats-ensemble-randomforest--xgboost)
5. [Analyse des Impacts](#analyse-des-impacts)
6. [Recommandations](#recommandations)

---

## Vue d'ensemble

### Contexte

Le projet de détection de fausses nouvelles utilise une approche **Part A/B split**:
- **Part A (80%)**: Entraînement des 3 branches (Style, Knowledge, Claims)
- **Part B (20%)**: Validation de la fusion entre branches

Cette étude compare les performances obtenues avec:
- **main.ipynb**: Utilise le **dataset complet (100%)** - 62,649 lignes
- **unified_main.ipynb**: Utilise **Part A seulement (80%)** - 45,255 lignes

### Questions clés

- Quel est l'impact de perdre 20% des données d'entraînement?
- La réduction affecte-t-elle la stabilité du modèle?
- Quel modèle (Random Forest vs XGBoost) bénéficie le plus des données supplémentaires?

---

## Méthodologie

### Données d'entraînement

| Source | Dataset | Lignes | % du total | Utilisé par |
|--------|---------|--------|-----------|------------|
| 100% complet | dataset.csv | 62,649 | 100% | main.ipynb |
| 80% (Part A) | dataset_partA.csv | 45,255 | 80% | unified_main.ipynb |
| **Différence** | — | **17,394** | **20%** | — |

### Pipeline identique pour les deux

1. **Phase 1**: data_extraction.py → dataset.csv (complet) ou dataset_partA.csv
2. **Phase 2**: feature_extraction.py → 20+ métriques stylométriques
3. **Phase 3**: split_data.py → Block A (60%), Block B (20%), Block C (20%)
4. **Phase 4**: fine_tunning.py → RoBERTa distilroberta-base
5. **Phase 5**: model_comp.py → Random Forest vs XGBoost

### Configuration d'évaluation

- **Block C (Test set)**: 
  - main.ipynb: 12,335 échantillons
  - unified_main.ipynb: 8,894 échantillons

- **Métriques**:
  - Accuracy (Principal)
  - F1 Score (Balance Precision/Recall)
  - ROC-AUC (Discriminabilité)
  - Log Loss (Calibration/Confiance)

---

## Résultats RoBERTa Baseline

### Résultats bruts

| Métrique | main.ipynb (100%) | unified_main.ipynb (80%) | Δ absolu | Δ relatif |
|----------|-------------------|--------------------------|----------|-----------|
| **Accuracy** | 92.44% | 89.90% | -2.54 pts | -2.75% ↓ |
| **F1 Score** | 92.21% | 89.51% | -2.70 pts | -2.93% ↓ |
| **ROC-AUC** | 98.52% | 98.17% | -0.35 pts | -0.36% ↓ |
| **Log Loss** | 15.30% | 21.37% | +6.07 pts | +39.6% ↑ |
| **Test samples** | 12,335 | 8,894 | -3,441 | -27.9% |

### Analyse par classe

#### main.ipynb (100% - 12,335 samples)
```
              precision    recall  f1-score   support
        FAKE       0.90      0.95      0.93      6,173
        TRUE       0.95      0.90      0.92      6,162
     accuracy                           0.92     12,335
```

#### unified_main.ipynb (80% - 8,894 samples)
```
              precision    recall  f1-score   support
        FAKE       0.96      0.85      0.90      4,875
        TRUE       0.84      0.95      0.90      4,019
     accuracy                           0.90      8,894
```

### Observations clés

1. **Précision FAKE aumenta** (90% → 96%): Modèle plus conservateur avec moins de données
2. **Recall FAKE diminue** (95% → 85%): Moins de faux positifs, mais données manquantes
3. **Trade-off Precision/Recall**: Impact asymétrique par classe
4. **Log Loss bien pire** (+39.6%): Calibration des probabilités moins fiable

---

## Résultats Ensemble (RandomForest + XGBoost)

### Phase 5: Optimisation et Comparaison

#### Random Forest

| Métrique | main.ipynb (100%) | unified_main.ipynb (80%) | Δ |
|----------|-------------------|--------------------------|---|
| **Accuracy** | 92.55% | 91.59% | -0.96 pts ↓ |
| **F1 Score** | 92.41% | 90.81% | -1.60 pts ↓ |
| **ROC-AUC** | 98.47% | 98.10% | -0.37 pts ↓ |
| **Log Loss** | 17.36% | 19.20% | +1.84 pts ↑ |
| **Meilleur?** | ✅ OUI | ❌ Non | — |

#### XGBoost

| Métrique | main.ipynb (100%) | unified_main.ipynb (80%) | Δ |
|----------|-------------------|--------------------------|---|
| **Accuracy** | 92.48% | 91.85% | -0.63 pts ↓ |
| **F1 Score** | 92.40% | 90.95% | -1.45 pts ↓ |
| **ROC-AUC** | 98.51% | 98.16% | -0.35 pts ↓ |
| **Log Loss** | 15.12% | 16.52% | +1.40 pts ↑ |
| **Meilleur?** | ❌ Non | ✅ OUI | — |

### Hyperparamètres découverts

#### Random Forest
- **main.ipynb**: `max_depth=10, min_samples_split=10, n_estimators=200`
- **unified_main.ipynb**: `max_depth=10, min_samples_split=10, min_samples_leaf=4, n_estimators=200`

**Différence**: unified_main.ipynb utilise `min_samples_leaf=4` (vs 1) → Données insuffisantes pour être trop granulaire

#### XGBoost
- **Identiques pour les deux**: `max_depth=3, learning_rate=0.01, subsample=0.8, n_estimators=500`

---

## Analyse des Impacts

### Impact 1: Dégradation de l'Accuracy

**Magnitude**: -0.6% à -2.7% selon le modèle

```
100% (main):    RoBERTa 92.44% → RF 92.55% → XGB 92.48%
                └─ Overhead ensemble minimal (+0.04% RF)

80% (unified):  RoBERTa 89.90% → RF 91.59% → XGB 91.85%
                └─ Overhead ensemble significatif (+1.69% RF, +1.95% XGB)
```

**Interprétation**: Avec moins de données, l'ensemble (RF+XGB) bénéficie davantage de la combinaison. RoBERTa seul souffre, mais l'ensemble rattrape partiellement.

### Impact 2: Dégradation de la Calibration (Log Loss)

**Magnitude**: +1.4% à +6.1%

```
RoBERTa:  15.30% → 21.37% (+6.07 pts, pire)
RF:       17.36% → 19.20% (+1.84 pts, modéré)
XGB:      15.12% → 16.52% (+1.40 pts, léger)
```

**Interprétation**: 
- Les probabilités du modèle sont moins fiables avec 80%
- RoBERTa souffre le plus (manque de données pour fine-tuning)
- XGBoost reste bien calibré grâce à sa nature probabiliste

### Impact 3: Sélection du meilleur modèle

**main.ipynb (100%)**: Random Forest champion
- Accuracy: 92.55% (vs 92.48%)
- Avantage: +0.07%

**unified_main.ipynb (80%)**: XGBoost champion
- Accuracy: 91.85% (vs 91.59%)
- Avantage: +0.26%

**Interprétation**: Avec 80% des données:
- Random Forest devient instable (overfitting à cause de moins de patterns)
- XGBoost bénéficie de sa régularisation native

### Impact 4: Stabilité du modèle

**Variance entre modèles**:
```
main.ipynb:  |92.55% (RF) - 92.48% (XGB)| = 0.07 pts (très proche)
unified:     |91.59% (RF) - 91.85% (XGB)| = 0.26 pts (écart 3.7x plus large)
```

**Conclusion**: Les 20% de données supplémentaires stabilisent les performances entre modèles rivaux.

---

## Recommandations

### 1. Pour la Production (Branche Style)

✅ **Recommandation**: Utiliser **100% des données** si possible

**Justification**:
- Gain d'accuracy: +0.6% à +2.7%
- Meilleure calibration: Log Loss -39.6%
- Stabilité: Gap RF/XGB réduit de 3.7x
- Coût: Temps d'entraînement +25%

### 2. Pour la Fusion (Part A/B)

✅ **Recommandation**: Garder le **Part A/B split (80/20)**

**Justification**:
- Part B (20%) garantit validation **sans data leakage**
- Fair evaluation de la fusion des 3 branches
- Sacrifice acceptable: -0.6% à -1.6% (F1) pour évaluation rigoureuse

**Architecture finale recommandée**:
```
Dataset complet (62,649)
    ├─ Part A (80%, 50,119) → Entraînement 3 branches
    │   ├─ Style Branch (dataset_partA.csv)
    │   ├─ Knowledge Branch (train_partA.jsonl)
    │   └─ Claim Detection (groundtruth_partA.csv)
    │
    └─ Part B (20%, 12,530) → Validation fusion
        └─ Ensemble voting sur predictions 3 branches
```

### 3. Sélection du modèle ensemble

**100% données (main.ipynb)**: Random Forest (92.55%)
- **Quand**: Priorité à l'accuracy
- **Coût**: Moins calibré (Log Loss 17.36%)

**80% données (unified_main.ipynb)**: XGBoost (91.85%)
- **Quand**: Priorité à la calibration + généralisation
- **Avantage**: Log Loss 16.52% (meilleur que RF)

### 4. Optimisations futures

- [ ] Augmentation de données (mixup, paraphrase)
- [ ] Hyperparamètre tuning plus agressif
- [ ] Stacking d'ensembles (RF + XGB + RoBERTa)
- [ ] Cross-validation stratifiée par source (LIAR, Twitter, UoVictoria)

---

## Tableau Résumé Exécutif

| Aspect | main.ipynb (100%) | unified_main.ipynb (80%) | Verdict |
|--------|-------------------|--------------------------|---------|
| **Données** | 62,649 lignes | 45,255 lignes | ➖ -20% |
| **RoBERTa Accuracy** | 92.44% | 89.90% | ❌ -2.54% |
| **Ensemble Best** | RF 92.55% | XGB 91.85% | ❌ -0.70% |
| **Log Loss** | 15.12% (XGB) | 16.52% (XGB) | ⚠️ +1.40% |
| **Stabilité** | GAP RF/XGB: 0.07% | GAP RF/XGB: 0.26% | ❌ Moins stable |
| **Meilleur modèle** | Random Forest | XGBoost | ⚠️ Changement |
| **Recommandation** | ✅ Utiliser 100% | ✅ Utiliser 80% pour Part A/B | Compromis optimal |

---

## Conclusion

### Trade-offs clés

1. **Accuracy vs Data Integrity**
   - 100% = +2.7% accuracy mais contamination possible Part A/B
   - 80% = Évaluation rigoureuse avec -2.7% accuracy

2. **Performance vs Fair Evaluation**
   - main.ipynb optimise pour accuracy pure
   - unified_main.ipynb optimise pour validation fusion sans biais

### Recommandation finale

**Phase d'entraînement (style_branch)**:
- Utiliser 100% dataset.csv → Best accuracy (92.55%)
- Modèle final: Random Forest

**Phase de fusion (fusion_branch)**:
- Utiliser 80% Part A pour entraîner 3 branches
- Valider sur Part B (20%) → Fusion sans leakage
- Expected gain fusion: +1-3% accuracy vs style seul

**Métrique clé**: F1-Score on Part B (fusion)
- Objectif: > 91% (≥ baseline style seul)
- Résultat attendu: 91-94% (combiner 3 perspectives)

---

## Annexes

### A. Fichiers sources

- [main.ipynb](main.ipynb) - Dataset 100%
- [unified_main.ipynb](unified_main.ipynb) - Part A (80%)
- [style_branch/feature_extraction.py](style_branch/feature_extraction.py) - Feature extraction
- [style_branch/model_comp.py](style_branch/model_comp.py) - Hyperparameter optimization

### B. Références de performance

- **Baseline académique**: 85-90% accuracy (LIAR dataset)
- **Notre performance**: 92.55% (100%), 91.85% (80%)
- **State-of-the-art**: ~93-94% (avec BERT large + ensemble)

### C. Prochaines étapes

1. Implémenter fusion_pipeline.py avec 3 stratégies de voting
2. Évaluer Part B pour déterminer gain fusion
3. Générer rapport d'impact fusion (esperado +1-3% accuracy)
4. Documentation architecture finale

---

**Document generated**: 11 avril 2026  
**Status**: ✅ Complet et prêt pour décisions architecturales

**Objective**: Compare model performance across full dataset (100%) vs Part A split (80%)

---

## Architecture

```
Project Structure:
├── style_branch/
│   ├── feature_extraction.py          ← 100% (dataset.csv)
│   ├── feature_extraction_partA.py    ← 80% (dataset_partA.csv)
│   ├── split_data.py
│   ├── fine_tunning.py
│   └── model_comp.py
│
└── knowledge_branch/
    ├── train_claim_detector.py          ← 100% (groundtruth.csv)
    ├── train_claim_detector_partA.py    ← 80% (groundtruth_partA.csv)
    ├── evaluate_pipeline.py             ← 100% (train.jsonl)
    └── evaluate_pipeline_partA.py       ← 80% (train_partA.jsonl)
```

---

## Dataset Sizes

### Style Branch

| File | Mode | Rows | Usage |
|------|------|------|-------|
| dataset.csv | 100% | 62,649 | feature_extraction.py |
| dataset_partA.csv | 80% | 45,255 | feature_extraction_partA.py |

### Knowledge Branch

| File | Mode | Rows | Usage |
|------|------|------|-------|
| groundtruth.csv | 100% | 1,032 | train_claim_detector.py |
| groundtruth_partA.csv | 80% | 780 | train_claim_detector_partA.py |
| train.jsonl (FEVER) | 100% | 145,449 | evaluate_pipeline.py |
| train_partA.jsonl (FEVER) | 80% | 81,962 | evaluate_pipeline_partA.py |

---

## Testing Protocol

### 1. Style Branch Comparison

**Baseline (100% data)**:
```bash
cd style_branch
python feature_extraction.py          # Load dataset.csv
python print_features.py
python split_data.py
python fine_tunning.py
python test_fine_tuned.py
python model_comp.py
python result_roberta.py
# Output: results/report_random_forest.txt, results/report_xgboost.txt
```

**Part A (80% data)**:
```bash
cd style_branch
python feature_extraction_partA.py    # Load dataset_partA.csv
python print_features.py
python split_data.py
python fine_tunning.py
python test_fine_tuned.py
python model_comp.py
python result_roberta.py
# Output: style_branch/results/*.txt (same files, different results)
```

**Action**: Modify [feature_extraction.py](style_branch/feature_extraction.py) line 21:
- For 100%: `input_file = "../data/dataset.csv"`
- For 80%: `input_file = "../data/splits/dataset_partA.csv"`

---

### 2. Knowledge Branch: Claim Detector

**Baseline (100% data)**:
```bash
cd knowledge_branch
python train_claim_detector.py        # Load groundtruth.csv (1,032 rows)
# Trains on 100% of ClaimBuster dataset
```

**Part A (80% data)**:
```bash
cd knowledge_branch
python train_claim_detector_partA.py  # Load groundtruth_partA.csv (780 rows)
# Trains on 80% of ClaimBuster dataset
```

**Files Generated**:
- 100% mode: `knowledge_branch/my_claim_model/` (100% trained)
- 80% mode: Same location but different weights (80% trained)

---

### 3. Knowledge Branch: Evidence Pipeline

**Baseline (100% data)**:
```bash
cd knowledge_branch
python evaluate_pipeline.py           # Load train.jsonl (145,449 rows)
# Evaluates on full FEVER dataset
```

**Part A (80% data)**:
```bash
cd knowledge_branch
python evaluate_pipeline_partA.py     # Load train_partA.jsonl (81,962 rows)
# Evaluates on 80% of FEVER dataset
```

**Output Files**:
- 100% mode: 
  - `results/confusion_matrix.png`
  - `results/evaluation_report.txt`
- 80% mode:
  - `results/confusion_matrix_partA.png`
  - `results/evaluation_report_partA.txt`

---

## Comparison Results (From COMPARISON_80_vs_100_PERCENT.md)

### Style Branch: RoBERTa Baseline

| Metric | 100% (12,335 samples) | 80% (8,894 samples) | Δ |
|--------|----------------------|-------------------|---|
| Accuracy | 92.44% | 89.90% | -2.54% ↓ |
| F1 Score | 92.21% | 89.51% | -2.70% ↓ |
| ROC-AUC | 98.52% | 98.17% | -0.35% ↓ |
| Log Loss | 15.30% | 21.37% | +6.07% ↑ |

### Style Branch: Best Ensemble

| Model | 100% | 80% | Δ |
|-------|------|-----|---|
| Random Forest | 92.55% | 91.59% | -0.96% ↓ |
| XGBoost | 92.48% | 91.85% | -0.63% ↓ |
| **Winner** | RF | XGB | — |

---

## Key Findings

1. **Accuracy Drop**: -0.6% to -2.7% with 20% less data
2. **Calibration Impact**: Log Loss increases +39.6% (less reliable probabilities)
3. **Stability**: Model selection changes with 80% (XGBoost beats RF)
   - 100%: Random Forest champion (92.55%)
   - 80%: XGBoost champion (91.85%)
4. **ROC-AUC Stable**: Discrimination power only -0.35% (robust metric)

---

## Recommendations

### For Production (Style Branch Only)
- ✅ Use **100% dataset** for best accuracy (92.55%)
- ✅ Model: **Random Forest**
- ❌ Sacrifice: None (full data available)

### For Part A/B Fusion
- ✅ Use **80% Part A** for fair validation
- ✅ Sacrifice: -0.96% accuracy (acceptable trade-off)
- ✅ Benefit: Part B (20%) unused for fusion testing

### Conclusion
The 2.7% accuracy loss is acceptable trade-off for rigorous Part A/B separation and fair fusion evaluation without data leakage.

---

## Scripts Checklist

### Style Branch
- [x] feature_extraction.py (100%)
- [x] feature_extraction_partA.py (80%)

### Knowledge Branch
- [x] train_claim_detector.py (100%)
- [x] train_claim_detector_partA.py (80%)
- [x] evaluate_pipeline.py (100%)
- [x] evaluate_pipeline_partA.py (80%)

---

## Next Steps

1. Run comparison tests systematically
2. Document results in `KNOWLEDGE_BRANCH_COMPARISON.md`
3. Update notebooks to reference correct scripts
4. Implement fusion pipeline on Part B validation set
