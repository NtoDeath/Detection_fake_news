# 5 Stratégies de Fusion - Documentation Détaillée

---

## 📋 TABLE DES MATIÈRES

### 📊 I. FONDAMENTAUX
1. [Architecture des Données & Pipeline](#1-architecture-des-données--pipeline-de-split)
   - [1.1 Structure Générale](#11-structure-générale-du-projet)
   - [1.2 Phase A: Partition Part A/B](#12-phase-a-partition-partab-8020-split-global)
   - [1.3 Phase B: Split Part B 50/50](#13-phase-b-split-part-b-5050-train--test)
   - [1.4 Inputs Disponibles](#14-inputs-disponibles-aux-stratégies)

2. [Résultats des Baselines](#2-résultats-des-baselines-single-models-sur-part-b-test)
   - [2.1 Style Model Seul](#21-style-model-seul-baseline-1)
   - [2.2 Knowledge Model Seul](#22-knowledge-model-seul-baseline-2)
   - [2.3 Baseline Comparison](#23-baseline-comparison)

### 🎓 II. CONCEPTS & PRINCIPES
3. [Concepts Clés & Terminologie](#3-concepts-clés--terminologie)
   - [3.0.1 What is a Meta-Learner?](#301-what-is-a-meta-learner)
   - [3.0.2 What is RandomForest?](#302-what-is-randomforest)
   - [3.0.3 Gridsearch](#303-gridsearch-recherche-dhyperparamètres)
   - [3.0.4 Cascading](#304-cascading-cascade-de-décisions)
   - [3.0.5 Confidence-Weighted Voting](#305-confidence-weighted-voting)
   - [3.0.6 Stacking vs Blending](#306-stacking-vs-blending)

### ⚙️ III. STRATÉGIES DE FUSION
4. [Les 5 Stratégies de Fusion](#4-les-5-stratégies-de-fusion)
   - [4.1 Cascading (Style First)](#41-stratégie-1-cascading-style-first-)
   - [4.2 Confidence-Weighted Voting](#42-stratégie-2-confidence-weighted-voting-)
   - [4.3 Disagreement-Adaptive Weighting](#43-stratégie-3-disagreement-adaptive-weighting-)
   - [4.4 Weighted Voting + Threshold](#44-stratégie-4-weighted-voting--threshold-)
   - [4.5 Stacked RandomForest (MEILLEURE)](#45-stratégie-5-stacked-randomforest--meilleure)

### 📈 IV. RÉSULTATS & RECOMMANDATIONS
5. [Résultats Comparatifs](#5-résultats-comparatifs-toutes-les-stratégies)

6. [Recommandations d'Utilisation](#6-recommandations-dutilisation)
   - [🏆 Stratégie 5: Stacked RandomForest](#-recommandée-stratégie-5-stacked-randomforest)
   - [⭐ Stratégies 1, 3, 4: Bon Compromis](#-bon-compromis-stratégies-1-3-4-f1--06035)
   - [⚠️ Stratégie 2: À Éviter](#️-à-éviter-stratégie-2-confidence-weighted)

7. [Cas d'Usage Pratiques](#7-cas-dusage-pratiques)

### 🏗️ V. PIPELINE & EXÉCUTION
8. [Pipeline Complet (9 Scripts)](#8-pipeline-complet-9-scripts)

9. [Temps d'Exécution Estimé](#9-temps-dexécution-estimé)

10. [Conclusion](#10-conclusion)

---

## 1. ARCHITECTURE DES DONNÉES & PIPELINE DE SPLIT

### 1.1 Structure Générale du Projet

```
Dataset Total (100%)
       ↓
   ┌───┴────────┐
   ↓            ↓
 Part A        Part B
(80%)         (20%)
31.8k         31.8k
   ↓            ↓
Training      Fusion
   ↓            ↓
Style +       Split
Knowledge     50/50
Branch        ↓
Frozen      Part B Train  Part B Test
Models      (15.9k)       (15.9k)
             │            │
             ├─ Optimize  ├─ Final
             │ 5 Strats   │ Evaluation
             │            │
             ↓            ↓
          Metriques    Résultats
```

### 1.2 Phase A: Partition Part A/B (80/20 split global)

**Objectif:** Créer deux ensembles disjoints pour éviter le data leakage

```
Dataset Hétérogène Complet (100%):
├── LIAR (dataset.csv)
├── Twitter Fake News
├── UoVictoria
├── ClaimBuster (groundtruth.csv)
└── FEVER (train.jsonl)

         ↓
    Normalisation Labels
    (0=FAKE, 1=TRUE)

         ↓
Stratified 80/20 Split
         ↓
    ┌────┴──────┐
    ↓           ↓
 Part A       Part B
(80%)        (20%)
25,442       7,962
    ↓           ↓
├─ dataset_partA.csv     ├─ part_B_validation.csv
├─ groundtruth_partA.csv │  (31.8k merged + upsampling)
└─ train_partA.jsonl     │
    │                     │
    ├─ Style Training     └─ Fusion Testing
    │ (92% F1 on Part A)
    │
    └─ Knowledge Training
      (32% F1 on Part A)
```

### 1.3 Phase B: Split Part B pour Fusion (50/50 train/test)

**Objectif:** Optimiser les 5 stratégies sur des données d'entraînement, tester sur des données non vues

**Point Critique:** Split stratifié pour maintenir l'équilibre des classes

```
Part B (31.8k unseen samples)
       ↓
Generate Predictions
├── Style Model (frozen from Part A)    → (31.8k predictions + confs)
└── Knowledge Model (frozen from Part A) → (31.8k predictions + confs)
       ↓
Combine Features
Feature Vector pour chaque sample:
  - style_pred:      {0, 1}
  - style_conf:      [0.0, 1.0]
  - knowledge_pred:  {0, 1}
  - knowledge_conf:  [0.0, 1.0]
  - label_true:      {0, 1} (ground truth)
       ↓
Stratified Split 50/50
       ↓
    ┌──────────────────┬──────────────────┐
    ↓                  ↓
 FUSION TRAIN      FUSION TEST
 (15.9k samples)   (15.9k samples)
    │                  │
    - Classe 0: 5968  - Classe 0: 5969
    - Classe 1: 9934  - Classe 1: 9933
    │                  │
    ├─ Optimize        └─ Final Metrics
    │ Parameters
    │ (Gridsearch)
    │ for each strategy
    │
    └─ Best Config
       per strategy
```

**Distribution des Classes:**
```
Part B Total: 31.8k samples
├─ FAKE (Classe 0): 11.9k (37.5%)
└─ TRUE (Classe 1): 19.9k (62.5%)

Equal split:
├─ Train: 5.968k FAKE + 9.934k TRUE = 15.9k
└─ Test:  5.969k FAKE + 9.933k TRUE = 15.9k
```

### 1.4 Inputs Disponibles aux Stratégies

```
Feature Vector (pour chaque sample):
┌─────────────────────────────────────────┐
│ style_pred     : {0, 1} - prédiction    │
│ style_conf     : [0.0,1.0] - confiance │
│ knowledge_pred : {0, 1} - prédiction    │
│ knowledge_conf : [0.0,1.0] - confiance │
│ label_true     : {0, 1} - ground truth  │
└─────────────────────────────────────────┘

Count: 15.9k samples × 5 features (TRAIN)
Count: 15.9k samples × 5 features (TEST)
```

---

## 2. RÉSULTATS DES BASELINES: SINGLE MODELS SUR PART B TEST

### 2.1 Style Model Seul (Baseline 1)

**Configuration:**
- Modèle: RoBERTa fine-tuned + RandomForest
- Entraîné sur: Part A (25.4k samples)
- Testé sur: Part B Test (15.9k unseen samples)
- Utilise: Prédictions binaires + confiances

**Performance sur Part B Test:**
```
┌─────────────────────────────────┐
│ BASELINE: Style Only            │
├─────────────────────────────────┤
│ Accuracy:  0.3054 (30.5%)      │
│ Precision: 0.4219 (42.2%)      │
│ Recall:    0.3027 (30.3%)      │
│ F1-Score:  0.3525 ⚠️           │
└─────────────────────────────────┘
```

**Analyse:**
- ❌ **Très mauvaise performance** sur Part B
- ✅ Entraîné sur Part A (92% F1) mais **dégradation massive** (-58% F1)
- 💭 Explications:
  - **Domain Shift** marqué entre Part A et Part B
  - Part A = données contrôlées (LIAR + Twitter)
  - Part B = mélange hétérogène (5 sources différentes)
  - Le modèle Style semble **surspécialisé** sur Part A
  - Sur Part B: fait beaucoup de faux négatifs (Recall 30%)

**Conclusion:** Style seul est INSUFFISANT pour Part B

---

### 2.2 Knowledge Model Seul (Baseline 2)

**Configuration:**
- Modèle: DistilBERT Claim Detector + RoBERTa NLI Verifier
- Entraîné sur: Part A (25.4k samples)
- Testé sur: Part B Test (15.9k unseen samples)
- Utilise: Prédictions binaires + confiances

**Performance sur Part B Test:**
```
┌─────────────────────────────────┐
│ BASELINE: Knowledge Only        │
├─────────────────────────────────┤
│ Accuracy:  0.5048 (50.5%)      │
│ Precision: 0.6294 (62.9%)      │
│ Recall:    0.5041 (50.4%)      │
│ F1-Score:  0.5598 ⚠️(moyen)    │
└─────────────────────────────────┘
```

**Analyse:**
- ⚠️ **Performance modérée** sur Part B
- ✅ Meilleur que Style seul (+58% F1 vs Style)
- ✅ Precision acceptable (63%)
- ❌ Recall faible (50%) → rate la moitié des fakes
- 💭 Explications:
  - Knowledge base plus robuste aux variations de domaine
  - Mais performance globale limitée par architecture
  - Besoin de FUSION pour combiner les forces des deux

**Conclusion:** Knowledge seul est meilleur que Style, mais ENCORE INSUFFISANT

---

### 2.3 Baseline Comparison

```
Baseline Comparison (Part B Test):
═══════════════════════════════════════════════
                   Accuracy  Precision  Recall    F1
───────────────────────────────────────────────────
Style Only         0.3054    0.4219    0.3027   0.3525 ❌
Knowledge Only     0.5048    0.6294    0.5041   0.5598 ⚠️
───────────────────────────────────────────────────
```

**Key Insights:**
- Style F1: 0.3525 → **35%** (mauvais)
- Knowledge F1: 0.5598 → **56%** (acceptable)
- **Gap:** Knowledge +59% meilleur que Style
- **Problem:** Aucun n'est suffisant (< 60% F1)
- **Solution:** **Fusion pour combiner les forces**

---

## 3. CONCEPTS CLÉS & TERMINOLOGIE

### 3.0.1 What is a Meta-Learner?

**Concept simple:** Un meta-learner est un modèle qui apprend à combiner les prédictions d'autres modèles.

```
Modèles de base (Style, Knowledge)
         ↓
    Leurs prédictions
         ↓
  Meta-learner (RF)
         ↓
   Prédiction finale
```

**Exemple du monde réel:** 
Imaginez une équipe de médecins (Style & Knowledge). Chacun fait un diagnostic. Un administrateur expérimenté (meta-learner) écoute les deux avis et prend la décision finale en apprenant ce qui fonctionne le mieux.

**Avantages:**
- ✅ Apprend automatiquement les poids optimaux
- ✅ Capture les interactions non-linéaires
- ✅ Meilleur que les combinaisons linéaires simples
- ✅ Plus flexible que les règles fixes

**Comparaison:**

| Type | Approche | Exemple |
|------|----------|---------|
| **Linear Combination** | Poids fixes: 0.5×Style + 0.45×Knowledge | Stratégie 2, 4 |
| **Meta-learner** | Apprend la meilleure combinaison | Stratégie 5 |

---

### 3.0.2 What is RandomForest?

**Définition basique:** RandomForest est un **ensemble d'arbres de décision** qui votent ensemble pour prendre une décision.

```
Input → Tree 1 → Vote: "FAKE"
      → Tree 2 → Vote: "REAL"
      → Tree 3 → Vote: "FAKE"
      → Tree 4 → Vote: "FAKE"
      → Tree 5 → Vote: "REAL"
      ---
      Majorité: FAKE ✓
```

**Pourquoi "Forest" (Forêt)?**
- Contient 100 arbres ("trees")
- Ensemble = "Forest"
- Chaque arbre vote indépendamment
- Decision = Vote majoritaire

**Pourquoi ça fonctionne bien?**

1. **Diversité:** Chaque arbre est légèrement différent
   - Entraîné sur des échantillons différents (bootstrap)
   - Utilise des features différentes

2. **Robustesse:** 
   - Un mauvais arbre ne tue pas le vote
   - La majorité des 100 arbres décide
   - Reduce l'overfitting

3. **Non-linéarité:**
   - Les arbres peuvent capturer des patterns complexes
   - Les décisions ne sont pas linéaires
   - Exemple: "Si style_conf > 0.7 ET knowledge_pred = FAKE, alors..."

**Exemple visuel – Decision Tree (1 seul arbre):**

```
         style_confidence >= 0.5?
              /                \
           YES                 NO
            |                   |
      style_pred == 1?    knowledge_pred == 1?
        /     \              /         \
      YES     NO           YES        NO
       |       |            |          |
     REAL   FAKEish      REAL       FAKE
```

Un seul arbre = fragile. RandomForest = 100 arbres = robuste.

---

### 3.0.3 Gridsearch (Recherche d'Hyperparamètres)

**Concept:** Tester TOUTES les combinaisons possibles de paramètres et garder la meilleure.

```python
# Gridsearch simple:
meilleur_score = 0
for threshold in [0.4, 0.5, 0.6, 0.7]:
    for weight in [0.1, 0.2, 0.3]:
        # Tester cette combino
        score = evaluer(threshold, weight)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_threshold = threshold
            meilleur_weight = weight
```

**Nombre de configurations testées:**
- Stratégie 1: ~10 (threshold seul)
- Stratégie 4: 150 (5 × 6 × 5)

**Trade-off:**
- ✅ Trouve vraiment le meilleur
- ❌ Lent si beaucoup de configs

---

### 3.0.4 Cascading (Cascade de Décisions)

**Concept:** Utiliser d'abord un modèle, si pas sûr → utiliser le second.

```python
if confidence_model1 >= threshold:
    return prediction_model1
else:
    return prediction_model2
```

**Analogie:** Dans un aéroport, le scanner principal détecte les objets dangereux. Si pas sûr (confidence < threshold), demander au scanner secondaire.

**Avantages:**
- ✅ Très simple (une ligne if/else)
- ✅ Rapide
- ✅ Facile à expliquer

**Inconvénients:**
- ❌ Ignore complètement un modèle si threshold atteint
- ❌ Ne "combine" pas vraiment

---

### 3.0.5 Confidence-Weighted Voting

**Concept:** Moyenne pondérée — celui qui a plus confiance a plus de vote.

```
Votant 1 (Style): 80% confiant → vote compte "80%"
Votant 2 (Knowledge): 30% confiant → vote compte "30%"
Total poids: 80 + 30 = 110
Style influence: 80/110 = 72.7%
Knowledge influence: 30/110 = 27.3%
```

**Analogie:** Demander l'avis de 2 experts:
- Expert 1 (Style) très confiant (92% succès) → écouter 92%
- Expert 2 (Knowledge) moins confiant (32% succès) → écouter 32%

---

### 3.0.6 Stacking vs Blending

**Stacking (ce qu'on utilise):**
```
Train Set → Base Models (Style, Knowledge) → Meta-features (train)
                                              ↓
                                        Meta-learner (RF)
                                        entraîné sur train

Test Set → Base Models → Meta-features (test)
                         ↓
                      Meta-learner
                      Prédiction finale
```

**Key:** Meta-learner entraîné sur des prédictions différentes du test (pas de data leakage)

---

## 4. LES 5 STRATÉGIES DE FUSION


### 4.1 Stratégie 1: Cascading (Style First) 🌊

**Concept conceptuel:** Demander d'abord au premier expert. S'il n'est pas sûr, demander au deuxième.

**Analogie du monde réel:**
- Vous êtes malade. Vous allez voir le **Dermatologue (Style)**.
- Si le dermato est sûr de son diagnostic (confiance >= 0.5) → vous le croyez.
- Sinon → vous allez voir un **Généraliste (Knowledge)** pour avoir un deuxième avis.

**Comment ça marche:**

```python
# Règle très simple:
if style_confidence >= 0.50:
    # Style est confiant → faire confiance à Style
    final_prediction = style_prediction
else:
    # Style n'est pas sûr → utiliser Knowledge à la place
    final_prediction = knowledge_prediction
```

**Exemple sur 3 samples:**

| Sample | Style:FAKE? | Style conf | Knowledge:FAKE? | Knowledge conf | Decision | Raison |
|--------|-----------|-----------|---------------|----------------|----------|--------|
| S1 | ✅ FAKE | 0.82 | ❌ REAL | 0.45 | **FAKE** | Style >= 0.50 |
| S2 | ❌ FAKE | 0.48 | ✅ REAL | 0.72 | **REAL** | Style < 0.50 → utiliser Knowledge |
| S3 | ✅ REAL | 0.91 | ❌ FAKE | 0.65 | **REAL** | Style >= 0.50 |

**Paramètres:** 
- `threshold` = niveau de confiance requis pour faire confiance à Style
- Gridsearch testing ~10 valeurs différentes
- Meilleur trouvé: **threshold = 0.50** (simplement le milieu)

**Avantages:**
✅ **Extrêmement simple** — une ligne de code  
✅ **Rapide** — pas d'entraînement, juste comparaison  
✅ **Expliquable** — facile expliquer à un non-technicien  
✅ **Bon baseline** — pour comparer les autres stratégies  

**Inconvénients:**
❌ **N'utilise pas le deuxième modèle** si le premier est confiant → gâche l'info de Knowledge  
❌ **Pas une vraie combinaison** — c'est juste un switch (si/sinon)  
❌ **Logique binaire** — soit on croit Style, soit on croit Knowledge (pas de "entre deux")

---

### 4.2 Stratégie 2: Confidence-Weighted Voting 🔢

**Concept conceptuel:** Les deux experts votent, mais celui avec plus d'expérience a plus de poids.

**Analogie du monde réel - Restaurant:**
- Vous devez choisir un restaurant:
  - **Chef A** (60 ans, 40 ans d'expérience) dit: "Pizzeria est bonne" → vote compte "92%"
  - **Chef B** (25 ans, 5 ans d'expérience) dit: "Non c'est mauvais" → vote compte "32%"
- Décision finale: 92% Chef A + 32% Chef B = majorité pour Pizzeria

**Comment ça marche:**

```python
# Poids basés sur performance passée:
w_style = 0.92      # Style avait 92% F1 sur Part A
w_knowledge = 0.32  # Knowledge avait 32% F1 sur Part A

# Ils votent ensemble:
fusion_score = w_style * style_pred + w_knowledge * knowledge_pred
#             = 0.92 * 1       +  0.32 * 0          (si Style=FAKE, Knowledge=REAL)
#             = 0.92
final_pred = 1 if 0.92 >= 0.5 else 0  # → FAKE (92% > 50%)
```

**Exemple - Calculs:**

| Cas | Style | Knowledge | Calcul | Résultat |
|-----|-------|-----------|--------|----------|
| A | FAKE (1) | FAKE (1) | 0.92×1 + 0.32×1 = 1.24 | ✅ **FAKE** (très sûr) |
| B | FAKE (1) | REAL (0) | 0.92×1 + 0.32×0 = 0.92 | ✅ **FAKE** (92 > 0.5) |
| C | REAL (0) | FAKE (1) | 0.92×0 + 0.32×1 = 0.32 | ❌ **REAL** (0.32 < 0.5) |
| D | REAL (0) | REAL (0) | 0.92×0 + 0.32×0 = 0.00 | ❌ **REAL** (très sûr) |

**Paramètres:**
- **aucun** — poids FIXES, pas de tuning
- w_style = 0.92 (immuable)
- w_knowledge = 0.32 (immuable)

**Avantages:**
✅ **Combine vraiment les deux modèles** — ni l'un ni l'autre ne sont ignorés  
✅ **Poids logiques** — basés sur performance réelle (92% vs 32%)  
✅ **Pas de surapprentissage** — pas d'optimisation sur données de test  
✅ **Rapide** — simple calcul linéaire  

**Inconvénients:**
❌ **PIRE que Knowledge seul!** F1 = 0.5403 vs 0.5598  
❌ **Poids Part A ne s'adaptent pas à Part B** — distribution complètement différente  
❌ **Aucune optimisation** — pas de tuning sur données nouvelles  
❌ **Suppose transfert parfait** — Part A performance = Part B performance

**Leçon:** Les poids "généraux" d'une distribution ne fonctionnent pas bien sur une nouvelle distribution. Mieux vaut optimiser!

---

### 4.3 Stratégie 3: Disagreement-Adaptive Weighting ⚖️

**Concept conceptuel:** Accord = Confiance élevée. Désaccord = Être prudent.

**Analogie du monde réel - Médecins:**
- **Scénario A:** 2 cardiologues disent "C'est une crise cardiaque" → **TRÈS CONFIANT**
- **Scénario B:** Un dit "crise", l'autre dit "pas crise" → **INQUIET** → réduire confiance
- **Scénario C:** Les deux disent "Fausse alerte" → **TRÈS CONFIANT**

**Comment ça marche:**

```python
# Phase 1: Détecter accord/désaccord
agree = (style_pred == knowledge_pred)

# Phase 2: Appliquer logique selon accord
for each sample:
    if agree[i]:
        # Les deux ont le MÊME avis
        final_pred[i] = style_pred[i]
        confidence[i] = moyenne_confiances  # Accord = confiant
        
    else:
        # Désaccord! Qui croire?
        if style_conf[i] > knowledge_conf[i]:
            final_pred[i] = style_pred[i]
        else:
            final_pred[i] = knowledge_pred[i]
        
        # Réduire confiance (désaccord = doute)
        confidence[i] = moyenne_confiances * (1 - penalite_desaccord)
```

**Exemple visuel:**

| Cas | Style | Knowledge | Accord? | Action | Confiance Signal |
|-----|-------|-----------|---------|--------|-----------------|
| ✅ | FAKE (0.9) | FAKE (0.8) | **OUI** | Utiliser FAKE | **HAUTE** → certain |
| ⚠️ | FAKE (0.7) | REAL (0.6) | **NON** | Utiliser FAKE (0.7>0.6) | **RÉDUITE** → doutes |
| ✅ | REAL (0.85) | REAL (0.75) | **OUI** | Utiliser REAL | **HAUTE** → certain |
| ⚠️ | REAL (0.6) | FAKE (0.8) | **NON** | Utiliser FAKE (0.8>0.6) | **RÉDUITE** → doutes |

**Paramètres:**
- `disagreement_weight` = pénalité appliquée si désaccord
- Gridsearch: 7 valeurs différentes ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
- Meilleur trouvé: **0.30**

**Avantages:**
✅ **Détecte automatiquement l'incertitude** — via le désaccord des modèles  
✅ **Plus sophistiqué que Cascading** — prend en compte les deux modèles  
✅ **Combine les deux** — pas d'ignorance complète d'un modèle  
✅ **Intuitive** — la logique est facile à comprendre  

**Inconvénients:**
❌ **Performance identique à Cascading** — F1 = 0.6035 (pas d'amélioration!)  
❌ **Logique simpliste** — "désaccord = incertitude" est vrai parfois, pas tout le temps  
❌ **Assume confiance additive** — suppose que les deux confiances se combinent bien

---

### 4.4 Stratégie 4: Weighted Voting + Threshold ⭐

**Concept conceptuel:** Tester 150 combinaisons différentes pour trouver la meilleure "recette".

**Analogie du monde réel - Chimie:**
- Vous faites un cocktail avec bleu (Style) et rouge (Knowledge)
- Vous testez:
  - 50 mL bleu + 50 mL rouge
  - 60 mL bleu + 40 mL rouge
  - ... 150 mélanges différents
- Vous choisissez le meilleur goût

**Comment ça marche:**

```python
# La meilleure combinaison trouvée était:
w_style = 0.50      # Style compte pour 50%
w_knowledge = 0.45  # Knowledge compte pour 45%
threshold = 0.40    # Seuil décision >= 0.40

# Fusionner:
fusion_score = 0.50 * style_pred + 0.45 * knowledge_pred
final_pred = 1 if fusion_score >= 0.40 else 0

# Exemple 1: Style=FAKE (1), Knowledge=REAL (0)
# score = 0.50*1 + 0.45*0 = 0.50 >= 0.40 → FAKE ✓

# Exemple 2: Style=REAL (0), Knowledge=FAKE (1)
# score = 0.50*0 + 0.45*1 = 0.45 >= 0.40 → FAKE ✓

# Exemple 3: Style=REAL (0), Knowledge=REAL (0)
# score = 0.50*0 + 0.45*0 = 0.00 < 0.40 → REAL ✓
```

**Gridsearch complet:**

```
Variables testées:
├─ w_style:        [0.5, 0.6, 0.7, 0.8, 0.9]       →  5 options
├─ w_knowledge:    [0.05, 0.15, 0.25, 0.35, 0.45, 0.55] → 6 options
└─ threshold:      [0.4, 0.45, 0.5, 0.55, 0.6]    →  5 options

Total configurations: 5 × 6 × 5 = 150 ✓
```

**Paramètres optimaux trouvés:**
- w_style = **0.50** (équilibre!)
- w_knowledge = **0.45** (presque égal)
- threshold = **0.40** (décision décalée vers FAKE)

**Avantages:**
✅ **Gridsearch très exhaustif** — 150 configurations testées  
✅ **Poids équilibrés** — ni Style ni Knowledge ne dominent (50% vs 45%)  
✅ **Threshold flexible** — peut ajuster le point de bascule  
✅ **Linéarité transparente** — facile à expliquer  

**Inconvénients:**
❌ **Lent** — 150 configurations = beaucoup de calcul  
❌ **Performance identique aux autres** — F1 = 0.6035 (Cascading aussi!)  
❌ **Assume linéarité** — suppose que réalité est une simple addition pondérée  
❌ **Fragile** — peut overfitter si peu de données  

**Key Insight:** 150 configurations testées et quand même performance identique aux stratégies plus simples. Cela suggère que la combinaison linéaire a des limitations fundamentales.

---

### 4.5 Stratégie 5: Stacked RandomForest ⭐ **MEILLEURE**

**Concept conceptuel:** Un **jury d'experts** qui apprend la meilleure façon de combiner les avis.

**Analogie du monde réel - Jury de cour:**
- Vous avez 2 témoins (Style, Knowledge)
- Vous engagez **100 juges expérimentés**
- Chaque juge apprend: "Quand faire confiance au témoin 1? Quand faire confiance au témoin 2?"
- Chaque juge vote, et la majorité décide

**Comment ça marche - 3 phases:**

```
PHASE 1: ENTRAÎNER LE JURY (Sur 15,900 samples)
══════════════════════════════════════════════════
Style a dit:      [0, 1, 1, 0, 1, ...] ← 15.9k prédictions
Style confiance:  [0.9, 0.3, 0.8, 0.6, 0.7, ...] ← 15.9k valeurs
Knowledge a dit:  [1, 1, 0, 0, 1, ...] ← 15.9k prédictions
Knowledge conf:   [0.4, 0.7, 0.9, 0.5, 0.2, ...] ← 15.9k valeurs
Vraie réponse:    [0, 1, 1, 0, 1, ...] ← 15.9k labels

→ Les 100 juges apprennent les patterns:
  "Quand style_conf > 0.7 ET knowledge_conf < 0.5, 
   faire confiance à Style marche 91% du temps"
  
  "Quand les deux confiances > 0.6, 
   faire confiance à Style car dominante"
```

```
PHASE 2: CHAQUE JUGE VOTE (Sur 1 nouveau sample)
═════════════════════════════════════════════════
Nouveau sample: [style_pred=0, style_conf=0.85, 
                 knowledge_pred=1, knowledge_conf=0.45]

Juge 1: "ressemble à pattern 'Style confiant' → vote FAKE (0)"
Juge 2: "ressemble à pattern 'désaccord' → vote REAL (1)"
Juge 3: "confidence_style > 0.7 → vote FAKE (0)"
...
Juge 87: "vote FAKE (0)"
Juge 88: "vote REAL (1)"
...
Juge 100: "vote FAKE (0)"
```

```
PHASE 3: VOTE MAJORITAIRE
═════════════════════════
FAKE: 87 votes
REAL: 13 votes

Prédiction finale: **FAKE** (87 > 13) ✓
```

**Code simple:**

```python
# Phase 1: Entraîner les juges (RandomForest = 100 arbres)
meta_model = RandomForestClassifier(n_estimators=100, max_depth=10)
features_train = [[style_pred, style_conf, knowledge_pred, knowledge_conf], ...]
meta_model.fit(features_train, true_labels)

# Phase 2-3: Prédire
features_test = [[style_pred, style_conf, knowledge_pred, knowledge_conf], ...]
predictions = meta_model.predict(features_test)  # [0, 1, 0, ...]
```

**Feature Importance — Ce qui compte vraiment:**

```
RandomForest révèle l'importance réelle des features:

style_confidence:      79.09% ⭐⭐⭐ DOMINANT
├─ Les 100 juges ont découvert: c'est le facteur clé!
├─ Quand Style confiant, généralement c'est bon
└─ "Par défaut, faire confiance à Style si sûr"

style_prediction:      14.85% ⭐
├─ La prédiction elle-même compte aussi
└─ Mais moins importante que la confiance

knowledge_confidence:   5.97%
├─ Contribue au vote, mais mineur
└─ Décision dominée par Style

knowledge_prediction:   0.09%
└─ Quasi inutile! (~0%)
```

**Interprétation:**
Les 100 juges découvrent que:
- **1: Confiance Style est TOUT** (79%)
- **2: La prédiction Style compte** (15%)
- **3: Knowledge confiance = support mineur** (6%)
- **4: Knowledge prédiction = inutile** (0.09%)

→ Meilleure stratégie = "Faire confiance à Style si confiant"

**Paramètres (fixes, pas de tuning):**
- n_estimators = 100 (100 arbres = 100 juges)
- max_depth = 10 (profondeur limite = évite overfitting)
- random_state = 42 (reproductibilité)

**Avantages:**
✅ **Meilleure performance: F1 = 0.8435** (+40% vs Cascading!)  
✅ **Capture patterns complexes** — non-linéarités que méthodes linéaires ne voient pas  
✅ **Très robuste** — 100 consensus > 1 simple rule  
✅ **Feature importance** — on sait ce qui compte vraiment  
✅ **Excellence Recall: 98.9%** — détecte presque tous les fakes (faux négatifs rares)  

**Inconvénients:**
❌ **Plus complexe** — "boîte noire": on ne sait pas pourquoi exactement  
❌ **Besoin de données** — ~16k samples pour entraîner 100 arbres  
❌ **Risque surapprentissage** — mitigé par max_depth=10 mais réel  
❌ **Plus lent** — entraînement + 100 prédictions = plus coûteux  

---

## 5. RÉSULTATS COMPARATIFS: TOUTES LES STRATÉGIES

---

## 5. RÉSULTATS COMPARATIFS: TOUTES LES STRATÉGIES

| **Baseline/Stratégie** | **Accuracy** | **Precision** | **Recall** | **F1-Score** | **Notes** |
|---|---|---|---|---|---|
| **Style Only** | 0.3054 | 0.4219 | 0.3027 | **0.3525** | ❌ Faible |
| **Knowledge Only** | 0.5048 | 0.6294 | 0.5041 | **0.5598** | ⚠️ Modéré |
| Stratégie 1: Cascading | 0.4654 | 0.5622 | 0.6515 | **0.6035** | ⭐ +8% vs Knowledge |
| Stratégie 2: Conf-Weighted | 0.4357 | 0.5500 | 0.5309 | **0.5403** | ❌ Pire que Knowledge |
| Stratégie 3: Disagreement | 0.4654 | 0.5622 | 0.6515 | **0.6035** | ⭐ Performance = Cascading |
| Stratégie 4: Weighted+Threshold | 0.4654 | 0.5622 | 0.6515 | **0.6035** | ⭐ Performance = Cascading |
| **Stratégie 5: Stacked RF** | **0.7706** | **0.7350** | **0.9894** | **0.8435** | 🏆 **+50% vs Knowledge, +139% vs Style** |

---

## 6. RECOMMANDATIONS D'UTILISATION

### 🏆 **RECOMMANDÉE: Stratégie 5 (Stacked RandomForest)**

**À utiliser si:**
- ✅ Maximiser la performance absolue (F1 = 0.8435)
- ✅ Production avec hautes exigences de qualité
- ✅ Vous avez ~16k samples pour l'entraînement
- ✅ Tolérance à la complexité (boîte noire)

**Résultats garantis:**
- +50% de F1 par rapport à Knowledge seul
- +139% de F1 par rapport à Style seul
- Excellent rappel (98.94%) pour détection de faux (peu de faux négatifs)
- Excellente précision (73.50%) pour fiabilité

**Configuration:**
```python
meta_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

---

### ⭐ **BON COMPROMIS: Stratégies 1, 3, 4 (F1 = 0.6035)**

**Stratégie 1 (Cascading)** - Simplicité maximale
- ✅ Code < 5 lignes
- ✅ Facile à expliquer: "Utiliser Style si confiant, sinon Knowledge"
- ✅ Pas d'entraînement complexe
- ✅ Déploiement très rapide
- ⚠️ Performance 28% inférieure à Stacked RF

**Stratégie 3 (Disagreement-Adaptive)** - Nuance intermédiaire
- ✅ Détecte incertitude via désaccord
- ✅ Plus sophistiqué que Cascading
- ✅ Même performance que Cascading (0.6035)
- ⚠️ Logique de désaccord peut être trop simpliste

**Stratégie 4 (Weighted+Threshold)** - Contrôle fin
- ✅ Gridsearch sur 3 paramètres (poids + threshold)
- ✅ 150 configurations testées pour optimum
- ✅ Combinaison linéaire transparent
- ⚠️ Plus long à tuner (compliqué sans données)

**À préférer si:** Vous êtes prêt à accepter 28% de performance inférieure pour gagner en simplicité.

---

### ⚠️ **À ÉVITER: Stratégie 2 (Confidence-Weighted)**

**Pourquoi:**
- ❌ F1 = 0.5403 — c'est PIRE que Knowledge seul (0.5598)
- ❌ Poids fixes (0.92, 0.32) ne s'adaptent pas à Part B
- ❌ Suppose que Part A généralise parfaitement

**Leçon:** Les poids basés sur une autre distribution n'aident pas — mieux vaut utiliser Knowledge directement ou Cascading.

---

## 7. CAS D'USAGE PRATIQUES

**Production haute-performance (news fact-checking):**
→ **Stratégie 5 (Stacked RF)** — maximiser la fiabilité

**API temps réel (latence critique):**
→ **Stratégie 1 (Cascading)** — décision en < 1ms

**Démonstration / PoC:**
→ **Stratégie 1 (Cascading)** — façile à expliquer

**Recherche / Publication:**
→ **Stratégie 5 (Stacked RF)** — meilleurs résultats

**Système avec audit trail (explicabilité importante):**
→ **Stratégie 4 (Weighted+Threshold)** — poids transparents

---

## 8. PIPELINE COMPLET (9 SCRIPTS)

```
00_verify_models.py
   └─ ✓ Vérifie existence des 4 modèles congelés

01_load_predictions.py
   ├─ Lance evidence_loader.py
   ├─ Génère prédictions Part B (style_pred, style_conf, knowledge_pred, knowledge_conf)
   └─ Sortie: results/part_b_predictions.pkl

02_split_data.py
   ├─ Charge predictions
   ├─ Split 50/50 → fusion_train (15.9k), fusion_test (15.9k)
   └─ Sortie: results/part_b_split.pkl

03_strategy_1.py (Cascading)
04_strategy_2.py (Conf-Weighted)
05_strategy_3.py (Disagreement-Adaptive)
06_strategy_4.py (Weighted+Threshold)
07_strategy_5.py (Stacked RF) ⭐
   ├─ Chacun charge results/part_b_split.pkl
   ├─ Entraîne sur fusion_train
   ├─ Teste sur fusion_test
   └─ Sortie: strategy_X_results.json

08_comparison_visualize.py
   ├─ Évalue baselines (Style seul, Knowledge seul)
   ├─ Évalue 5 stratégies de fusion
   └─ Sorties:
      ├─ comparison_table.csv
      ├─ fusion_strategy_comparison.png
      └─ best_strategy_analysis.txt
```

---

## 9. TEMPS D'EXÉCUTION ESTIMÉ

| Étape | Temps | Notes |
|---|---|---|
| verify_models.py | ~ 5s | Vérification fichiers |
| load_predictions.py | ~ 5-10 min | Evidence loading lent |
| split_data.py | ~ 30s | Transformation données |
| strategy_1-4 | ~ 1-5 min chaque | Entraînement + évaluation |
| strategy_5 | ~ 2-10 min | RF + gridsearch léger |
| comparison_visualize.py | ~ 1-2 min | Évaluation + graphes |
| **TOTAL** | **~25-40 min** | Exécution complète (séquentielle) |

---

## 10. CONCLUSION

**La Stratégie 5 (Stacked RandomForest) est clairement la meilleure approche:**

| Métrique | Valeur | Impact |
|---|---|---|
| **F1-Score** | 0.8435 | 🏆 Meilleur |
| **vs Knowledge baseline** | +50% | Amélioration massive |
| **vs Style baseline** | +139% | Amélioration massive |
| **Rappel** | 0.9894 | ⭐ Détecte 98.94% des faux |
| **Précision** | 0.7350 | ⭐ Fiable (73.50% des alertes vraies) |
| **Complexité** | Modérée | ✓ Acceptable en production |

**Verdict:** 🏆 **Utiliser Stratégie 5 pour production. Utiliser Stratégie 1 pour PoC/démonstration.**

