# 🚀 PLAN D'IMPLÉMENTATION - FUSION PIPELINE
**Date:** 11 Avril 2026  
**Objectif:** Implémenter le système de fusion multi-branches pour validation sur Part B  
**Audience:** Agent autonome de codage  

---

## 1. CONTEXTE & OBJECTIFS

### 1.1 Résumé du projet

Le pipeline de détection de fausses nouvelles comporte **3 branches indépendantes** entraînées sur **Part A (80% des données)**:

- **Style-based Branch**: Classifie via RoBERTa + RandomForest/XGBoost sur caractéristiques stylométriques
- **Knowledge-based Claim Detection**: Détecte si le texte contient des claims vérifiables (DistilBERT fine-tuné)
- **Knowledge-based Verification**: Vérifie les claims via NLI et récupération d'evidence (FEVER pipeline)

**Objectif de la Fusion**: Combiner les 3 prédictions indépendantes sur **Part B (20% validation)** pour:
- Valider que la fusion améliore la performance globale
- Documenter la stratégie d'ensemble
- Générer un rapport complet d'évaluation

### 1.2 Données

**Input:**
- `data/splits/part_B_validation.csv` (31,804 rows)
  - Colonnes: `text`, `label` (0=FAKE, 1=TRUE), `source` (DATA/GROUNDTRUTH/FEVER)
  - 37.5% FAKE, 62.5% TRUE (distribution équilibrée avec Part A)

**Modèles entraînés (Part A):**
- `style_branch/results/best_model.pkl` - Classifieur ensemble sélectionné (RF ou XGB)
- `style_branch/roberta_fine_tunned/` - RoBERTa fine-tuné
- `knowledge_branch/claim_detector_model/` - DistilBERT pour claim detection
- `knowledge_branch/my_claim_model/` - Modèle alternatif claim detection
- `knowledge_branch/full_pipeline.py` - Pipeline de vérification complète

---

## 2. ARCHITECTURE DÉTAILLÉE

### 2.1 Structure du project

```
fusion_branch/
├── fusion_pipeline.py              # Script principal (CRÉER)
├── classifier.py                   # Existing helper (vérifier/adapter)
├── fusion_strategies.py            # Différentes stratégies de fusion (CRÉER)
├── evidence_loader.py              # Load evidence + models (CRÉER)
├── config.py                       # Configuration centralisée (CRÉER)
└── results/
    ├── fusion_predictions.csv      # Output: predictions + confidence
    ├── fusion_report.txt           # Output: metrics complets
    ├── confusion_matrix.png        # Output: visualization
    ├── confusion_matrix_data.json  # Output: raw data
    ├── per_source_analysis.txt     # Output: breakdown by source
    ├── fusion_strategy.txt         # Output: which strategy used
    ├── calibration_curve.png       # Output: confidence calibration
    └── detailed_errors.csv         # Output: misclassifications analysis
```

### 2.2 Flux de données

```
Part B Data (31.8k samples)
    ↓
    ├─→ Route 1: STYLE
    │   ├─ Load: best_model.pkl
    │   ├─ Extract features (20+ stylometric)
    │   ├─ Predict: style_pred (0/1), style_prob [0-1]
    │   └─ Output: style_predictions (31.8k x 2)
    │
    ├─→ Route 2: CLAIM DETECTION
    │   ├─ Load: claim_detector_model/
    │   ├─ Predict: is_claim (boolean), claim_confidence
    │   └─ Filter: Only texts with claim_confidence > 0.3
    │
    ├─→ Route 3: VERIFICATION (for detected claims only)
    │   ├─ Load: full_pipeline components
    │   ├─ Extract entities
    │   ├─ Retrieve evidence
    │   ├─ Run NLI
    │   ├─ Predict: verify_pred, verify_confidence
    │   └─ Output: verify_predictions (subset x 2)
    │
    └─→ FUSION
        ├─ Strategy: Weighted voting / Confidence ensemble / Cascading
        ├─ Combine: style, claim, verify predictions
        ├─ Output: fusion_pred (0/1), fusion_confidence
        └─ Compare vs Part B ground truth
```

---

## 3. SPÉCIFICATIONS TECHNIQUES

### 3.1 Configuration globale (`fusion_branch/config.py`)

```python
# Paths
PART_B_PATH = Path("../data/splits/part_B_validation.csv")
STYLE_MODEL_PATH = Path("../style_branch/results/best_model.pkl")
ROBERTA_PATH = Path("../style_branch/roberta_fine_tunned/")
CLAIM_DETECTOR_PATH = Path("../knowledge_branch/claim_detector_model/")
FEATURE_EXTRACTOR_PATH = Path("../style_branch/style_extractor.py")

# Thresholds
CLAIM_THRESHOLD = 0.3              # Min confidence to consider text as claim
ENTAILMENT_THRESHOLD = 0.7         # Min confidence for SUPPORTS verdict
NLI_ABSTAIN_THRESHOLD = 0.3        # Max confidence to abstain

# Fusion strategy
FUSION_STRATEGY = "weighted_voting"  # Options: weighted_voting, confidence_ensemble, cascading
# Weights for weighted voting (sum = 1.0)
STYLE_WEIGHT = 0.50
CLAIM_WEIGHT = 0.25
VERIFY_WEIGHT = 0.25

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
LOG_LEVEL = logging.INFO
VERBOSE = True
```

### 3.2 Module: Evidence Loader (`fusion_branch/evidence_loader.py`)

**Purpose:** Charger tous les modèles et préparateurs nécessaires

**Classe:** `EvidenceLoader`

```python
class EvidenceLoader:
    def __init__(self, config: Config):
        """Initialize and load all models/components"""
        self.config = config
        self.logger = setup_logger(__name__)
        
    def load_style_model(self) -> dict:
        """
        Load pre-trained style classifier
        
        Returns:
            {
                'model': sklearn classifier,
                'feature_extractor': StyleExtractor instance,
                'input_shape': int,
                'output_classes': 2
            }
        """
        pass
    
    def load_claim_detector(self) -> dict:
        """
        Load claim detection model (DistilBERT)
        
        Returns:
            {
                'model': HuggingFace model,
                'tokenizer': HuggingFace tokenizer,
                'device': torch.device
            }
        """
        pass
    
    def load_verification_pipeline(self) -> dict:
        """
        Load NLI verification model + entity extractor + evidence retriever
        
        Returns:
            {
                'nli_model': model,
                'nli_tokenizer': tokenizer,
                'entity_extractor': SpaCy/stanza model,
                'evidence_db': loaded evidence database (FEVER),
                'device': torch.device
            }
        """
        pass
    
    def verify_models(self) -> bool:
        """Verify all models loaded successfully and are inference-ready"""
        pass
```

### 3.3 Module: Fusion Strategies (`fusion_branch/fusion_strategies.py`)

**Purpose:** Implémenter différentes stratégies de fusion

```python
class FusionStrategy:
    """Abstract base class for fusion strategies"""
    
    @abstractmethod
    def fuse(self, 
             style_pred: np.ndarray,      # shape: (N,)
             style_prob: np.ndarray,      # shape: (N, 2) or (N,)
             claim_pred: np.ndarray,      # shape: (N,) or None if not available
             claim_prob: np.ndarray,      # shape: (N,) or None
             verify_pred: np.ndarray,     # shape: (M,) M <= N
             verify_prob: np.ndarray,     # shape: (M,) or (M, 3)
             claim_detected: np.ndarray   # shape: (N,) boolean mask
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse predictions from 3 branches
        
        Returns:
            (fusion_pred, fusion_confidence)
            - fusion_pred: (N,) binary predictions {0, 1}
            - fusion_confidence: (N,) confidence scores [0, 1]
        """
        pass


class WeightedVoting(FusionStrategy):
    """Weighted majority voting with configurable weights"""
    
    def __init__(self, 
                 style_weight: float = 0.5,
                 claim_weight: float = 0.25,
                 verify_weight: float = 0.25):
        """
        Args:
            style_weight: Weight for style branch (0-1)
            claim_weight: Weight for claim branch (0-1)
            verify_weight: Weight for verification branch (0-1)
            
        Note: Weights should sum to 1.0 for interpretability
        """
        self.weights = {
            'style': style_weight,
            'claim': claim_weight,
            'verify': verify_weight
        }
    
    def fuse(self, ...) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted voting:
        - For each sample, compute: vote = style_pred * w1 + claim_pred * w2 + verify_pred * w3
        - If vote > 0.5: predict 1 (TRUE), else 0 (FAKE)
        - confidence = abs(vote - 0.5) * 2  # distance from decision boundary
        """
        pass


class ConfidenceEnsemble(FusionStrategy):
    """Confidence-based weighted ensemble"""
    
    def fuse(self, ...) -> Tuple[np.ndarray, np.ndarray]:
        """
        Confidence-weighted:
        - Dynamic weighting based on model confidence
        - For each sample, normalize confidences across available predictions
        - Weighted sum: pred = (pred1*conf1 + pred2*conf2 + ...) / sum(confidences)
        - Output confidence = max(confidence across branches)
        """
        pass


class CascadingStrategy(FusionStrategy):
    """Hierarchical decision making"""
    
    def fuse(self, ...) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cascading (hierarchical):
        1. If claim_detected AND verify_confidence > threshold:
           → Use verify prediction (60% weight)
           → Combine with style prediction (40% weight)
        2. Else if claim_detected BUT verify_confidence low:
           → Use style prediction + claim prediction equally
        3. Else (no claim detected):
           → Use only style prediction
        """
        pass
```

### 3.4 Pipeline Principal (`fusion_branch/fusion_pipeline.py`)

**Purpose:** Orchestrer tout le pipeline de fusion

```python
class FusionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.loader = EvidenceLoader(config)
        
    def run(self) -> Dict[str, Any]:
        """
        Main execution flow
        
        Returns:
            {
                'predictions': (N,) array of predictions,
                'confidence': (N,) array of confidence scores,
                'ground_truth': (N,) array of ground truth labels,
                'metrics': evaluation metrics dict,
                'report_path': path to detailed report,
                'predictions_csv_path': path to predictions CSV
            }
        """
        
        # Step 1: Load data
        part_B = self._load_part_b()
        
        # Step 2: Load all models
        self._load_models()
        
        # Step 3: Style-based predictions
        style_pred, style_prob = self._get_style_predictions(part_B)
        
        # Step 4: Claim detection
        claim_pred, claim_prob, claim_detected = self._get_claim_predictions(part_B)
        
        # Step 5: Verification (only for detected claims)
        verify_pred, verify_prob = self._get_verification_predictions(part_B, claim_detected)
        
        # Step 6: Fusion
        fusion_pred, fusion_conf = self._fuse_predictions(
            style_pred, style_prob,
            claim_pred, claim_prob,
            verify_pred, verify_prob,
            claim_detected
        )
        
        # Step 7: Evaluation
        metrics = self._evaluate(fusion_pred, fusion_conf, part_B['label'])
        
        # Step 8: Generate reports
        report_paths = self._generate_reports(
            part_B, fusion_pred, fusion_conf, metrics
        )
        
        return {
            'predictions': fusion_pred,
            'confidence': fusion_conf,
            'ground_truth': part_B['label'].values,
            'metrics': metrics,
            'reports': report_paths
        }
    
    def _load_part_b(self) -> pd.DataFrame:
        """Load and validate Part B data"""
        pass
    
    def _load_models(self) -> None:
        """Load all 3 branches models"""
        pass
    
    def _get_style_predictions(self, part_B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from style branch
        
        Returns:
            style_pred: (N,) binary predictions
            style_prob: (N, 2) or (N,) probability array
        """
        pass
    
    def _get_claim_predictions(self, part_B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get claim detection predictions
        
        Returns:
            claim_pred: (N,) binary (is_claim or not)
            claim_prob: (N,) confidence scores
            claim_detected: (N,) boolean mask (confidence > threshold)
        """
        pass
    
    def _get_verification_predictions(self, part_B: pd.DataFrame, claim_detected: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get verification predictions (only for detected claims)
        
        Returns:
            verify_pred: (M,) predictions where M <= N (only detected claims)
            verify_prob: (M,) confidence scores
        """
        pass
    
    def _fuse_predictions(self, ...) -> Tuple[np.ndarray, np.ndarray]:
        """Combine 3 branches using configured strategy"""
        pass
    
    def _evaluate(self, pred: np.ndarray, conf: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Returns:
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1': float,
                'roc_auc': float,
                'specificity': float,
                'confusion_matrix': [[TN, FP], [FN, TP]],
                'accuracy_by_source': {source: accuracy},
                'f1_by_source': {source: f1}
            }
        """
        pass
    
    def _generate_reports(self, part_B: pd.DataFrame, pred: np.ndarray, conf: np.ndarray, metrics: dict) -> Dict[str, str]:
        """
        Generate all output reports
        
        Returns:
            {
                'predictions_csv': path to CSV,
                'metrics_txt': path to text report,
                'confusion_matrix_png': path to figure,
                'source_analysis_txt': path to detailed analysis
            }
        """
        pass
```

---

## 4. FICHIERS DE SORTIE DÉTAILLÉS

### 4.1 `fusion_results/fusion_predictions.csv`
```
text,label_true,prediction_fusion,confidence_fusion,style_pred,style_conf,claim_detected,claim_conf,verify_pred,verify_conf,source
"The president...",1,1,0.92,1,0.88,True,0.75,1,0.95,DATA
"Breaking news...",0,0,0.81,0,0.79,False,0.25,NaN,NaN,FEVER
...
```

### 4.2 `fusion_results/fusion_report.txt`
```
============================================================
FUSION PIPELINE EVALUATION REPORT
============================================================

Strategy Used: weighted_voting
Weights: Style=0.50, Claim=0.25, Verify=0.25

Part B Dataset: 31,804 samples
- FAKE: 11,937 (37.5%)
- TRUE: 19,867 (62.5%)

GLOBAL METRICS
==============
Accuracy:  0.8124
Precision: 0.7945
Recall:    0.8412
F1-score:  0.8174
ROC-AUC:   0.8567
Specificity: 0.7654

Confusion Matrix:
                Predicted FAKE  Predicted TRUE
Actual FAKE         9123            2814
Actual TRUE         2154           17713

PER-SOURCE METRICS
==================

SOURCE: DATA (LIAR+Twitter+UoVictoria)
  Samples: 11,285 (35.5%)
  Accuracy: 0.8234
  F1-score: 0.8301
  
SOURCE: GROUNDTRUTH (ClaimBuster)
  Samples: 189 (0.6%)
  Accuracy: 0.7777
  F1-score: 0.7812
  
SOURCE: FEVER (Evidence-based)
  Samples: 20,330 (63.9%)
  Accuracy: 0.8112
  F1-score: 0.8156

BRANCH CONTRIBUTIONS
====================

Style-only baseline: Accuracy=0.7812, F1=0.7945
Fusion prediction:   Accuracy=0.8124, F1=0.8174
Improvement: +3.2% accuracy, +2.3% F1

============================================================
```

### 4.3 `fusion_results/per_source_analysis.txt`
```
PER-SOURCE DETAILED ANALYSIS
=============================

### DATA SOURCE (LIAR+Twitter+UoVictoria) - 11,285 samples ###

Style Branch Performance:
  Accuracy: 0.8156
  Precision: 0.7834
  Recall: 0.8401
  F1: 0.8109

Claim Detection:
  Claims detected: 8,945 (79.2%)
  Average confidence: 0.72

Verification Performance (on claimed texts):
  Accuracy: 0.8023
  F1: 0.8034

Fusion Improvement:
  +0.78% vs Style baseline
  +1.02% vs Claim-only

Misclassifications: 1,943
  - Types: Style confident but wrong (45%), Claim-verify disagreement (32%), etc.

---

### GROUNDTRUTH SOURCE (ClaimBuster) - 189 samples ###
[More detailed breakdown per source]

---

### FEVER SOURCE (Evidence-based) - 20,330 samples ###
[More detailed breakdown per source]

============================================================
```

### 4.4 `fusion_results/fusion_strategy.txt`
```
FUSION STRATEGY CONFIGURATION
==============================

Strategy Name: weighted_voting
Description: Weighted majority voting with per-branch weights

Weights:
  Style branch:        0.50 (50%)
  Claim branch:        0.25 (25%)
  Verification branch: 0.25 (25%)

Decision Process:
  For each sample:
    1. Compute weighted sum: vote = style*0.5 + claim*0.25 + verify*0.25
    2. If vote >= 0.5: predict 1 (TRUE)
    3. Else: predict 0 (FAKE)
    4. Confidence = |vote - 0.5| * 2

Special Cases:
  - If claim NOT detected: only use style prediction
  - If verify confidence < 0.3: weight verify at 0.1 instead of 0.25
  
Thresholds:
  Claim detection threshold: 0.30
  Entailment threshold: 0.70
  
Performance:
  Selected because: Best F1 on validation (0.8174 vs 0.8089 confidence_ensemble)
  
============================================================
```

---

## 5. TESTS & VALIDATION

### 5.1 Tests unitaires à implémenter

```python
# tests/test_fusion_strategies.py
def test_weighted_voting_basic():
    """Test basic weighted voting with simple inputs"""
    pass

def test_weighted_voting_edge_cases():
    """Test edge cases: all 1s, all 0s, mixed confidences"""
    pass

def test_confidence_ensemble():
    """Test confidence-based ensemble"""
    pass

def test_cascading_strategy():
    """Test hierarchical cascading"""
    pass

# tests/test_evidence_loader.py
def test_load_style_model():
    """Verify style model loads and runs"""
    pass

def test_load_claim_detector():
    """Verify claim detector loads and runs"""
    pass

def test_load_verification_pipeline():
    """Verify verification pipeline loads"""
    pass

# tests/test_pipeline.py
def test_pipeline_end_to_end():
    """Full pipeline execution on small sample"""
    pass

def test_output_format_validation():
    """Verify all output CSVs/JSONs have correct structure"""
    pass
```

### 5.2 Checkpoints de validation

- [ ] Part B data loads correctly (31,804 rows)
- [ ] All 3 models load without errors
- [ ] Style predictions generated for all samples
- [ ] Claim detection works (outputs in [0,1])
- [ ] Verification runs only on detected claims
- [ ] Fusion produces binary predictions + confidence
- [ ] Metrics computed correctly
- [ ] All output files generated
- [ ] No NaN/Inf in predictions
- [ ] Ground truth matches Part B labels

---

## 6. INTÉGRATION NOTEBOOK

### 6.1 Nouvelle cellule pour `unified_main.ipynb` (Phase 6)

```python
# Phase 6: Fusion Validation
%cd fusion_branch
!python fusion_pipeline.py
%cd ..

# Display results
import pandas as pd
import json

results = pd.read_csv('fusion_branch/results/fusion_predictions.csv')
print(f"✓ Fusion complete: {len(results)} predictions")
print(f"  Accuracy: {(results['prediction_fusion'] == results['label_true']).mean():.4f}")
print(f"  Confidence (mean): {results['confidence_fusion'].mean():.4f}")
```

---

## 7. STRUCTURE DÉTAILLÉE DES FONCTIONS

### 7.1 Extraction des features stylométriques

**Input:** Texte brut  
**Output:** Array de 20+ features (voir `style_branch/style_extractor.py`)

**Features typiques:**
- Longueur: avg_word_length, avg_sentence_length, text_length
- Vocabulaire: unique_words_ratio, type_token_ratio
- Complexité: flesch_kincaid_grade, gunning_fog_index
- Ponctuation: exclamation_ratio, quotation_ratio
- Sentiment: vader_negative, vader_positive, vader_neutral
- Véracité: capslock_ratio, special_char_ratio, url_count

### 7.2 Détection de claims

**Input:** Texte  
**Model:** DistilBERT fine-tuné sur ClaimBuster  
**Output:** 
- `is_claim`: 0/1 (binary)
- `confidence`: [0,1] (softmax probability)

**Threshold:** claim_confidence > 0.30 pour "detected"

### 7.3 Vérification via NLI

**Input:** claim text + retrieved evidence  
**Pipeline:**
1. Entity extraction (SpaCy): Extract subject, predicate, object
2. Evidence retrieval (FEVER DB): Find supporting/refuting evidence
3. NLI model: Compute entailment probability
4. Mapping: 
   - entailment > 0.7 → SUPPORTS (TRUE)
   - contradiction > 0.7 → REFUTES (FAKE)
   - neutral → ABSTAIN

**Output:**
- `verify_pred`: 0/1 or ABSTAIN
- `verify_confidence`: [0,1]

---

## 8. GESTION D'ERREURS

### 8.1 Erreurs critiques (stop execution)

- Model loading fails
- Part B file not found
- CUDA OOM (fallback to CPU)
- Invalid predictions (NaN/Inf)

### 8.2 Erreurs non-critiques (log + skip)

- Feature extraction fails for single sample → skip sample, log error
- Evidence retrieval empty → use style only for that sample
- Claim confidence scores out of [0,1] → clip to bounds

### 8.3 Logging

```python
logger.info(f"Loading {len(part_B)} samples from Part B")
logger.warning(f"Skipping {n_failed} samples due to feature extraction error")
logger.error(f"Model loading failed: {model_path}")
```

---

## 9. DELIVERABLES FINAUX

### 9.1 Code deliverables

- [ ] `fusion_branch/fusion_pipeline.py` (main script)
- [ ] `fusion_branch/fusion_strategies.py` (all strategies)
- [ ] `fusion_branch/evidence_loader.py` (model loading)
- [ ] `fusion_branch/config.py` (configuration)
- [ ] `tests/test_*.py` (unit tests)

### 9.2 Output deliverables

- [ ] `fusion_branch/results/fusion_predictions.csv` (31.8k rows)
- [ ] `fusion_branch/results/fusion_report.txt` (metrics + analysis)
- [ ] `fusion_branch/results/confusion_matrix.png` (visualization)
- [ ] `fusion_branch/results/per_source_analysis.txt` (detailed breakdown)
- [ ] `fusion_branch/results/fusion_strategy.txt` (strategy documentation)

### 9.3 Documentation deliverables

- [ ] Source code comments (docstrings for all functions)
- [ ] README.md in fusion_branch/
- [ ] IMPLEMENTATION_LOG.md (what was implemented)

---

## 10. TIMELINE ESTIMATION

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1 | Setup structure + config | 30 min | Create files, organize imports |
| 2 | EvidenceLoader implementation | 1 hour | Load 3 models, verify inference |
| 3 | FusionStrategy implementations | 1.5 hours | 3 strategies total |
| 4 | Main pipeline orchestration | 1 hour | Integrate all pieces |
| 5 | Evaluation + metrics | 45 min | Compute all metrics |
| 6 | Report generation | 45 min | Format output files |
| 7 | Testing + debugging | 1 hour | Unit tests + integration |
| 8 | Notebook integration | 15 min | Add Phase 6 cell |
| **Total** | | **~6.5 hours** | Single developer |

---

## 11. NOTES IMPORTANTES

1. **Pas de modification des Part A models**: Les 3 branches (style, claim, verify) doivent rester figées
2. **Part B est untouched**: Aucun entraînement, seulement évaluation
3. **Réplicabilité**: Seed=42 pour tous les random operations
4. **Performance**: Part B = 31.8k samples → ~2-3 min per branch inference
5. **Failure modes à éviter**: 
   - Ne pas normaliser différemment Part B vs Part A
   - Ne pas modifier labels Part B pour "faire passer" la fusion
   - Ne pas cherry-pick le meilleur strategy post-hoc

---

**FIN DU PLAN D'IMPLÉMENTATION**
