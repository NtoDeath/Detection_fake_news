"""
Fusion configuration centralisée - 5 Stratégies

Configuration pour 5 stratégies de fusion indépendantes:
1. Cascading (Style First)
2. Confidence-Weighted
3. Disagreement-Adaptive
4. Weighted Voting + Threshold (⭐ MEILLEUR)
5. Stacked RandomForest (⭐ ALTERNATIF)

12 Avril 2026
"""

import numpy as np
from pathlib import Path

# Torch optionnel
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# ==================== PATHS ====================
PART_B_PATH = Path("../data/splits/part_B_validation.csv")
STYLE_MODEL_PATH = Path("../style_branch/results/best_model.pkl")
STYLE_ROBERTA_PATH = Path("../style_branch/roberta_fine_tunned")
KNOWLEDGE_CLAIM_MODEL_PATH = Path("../knowledge_branch/claim_detector_model")
KNOWLEDGE_NLI_MODEL_PATH = Path("../knowledge_branch/my_claim_model")

# Dictionnaire PATHS pour evidence_loader
PATHS = {
    'part_b_data': PART_B_PATH,
    'style_model': STYLE_MODEL_PATH,
    'style_roberta': STYLE_ROBERTA_PATH,
    'knowledge_claim': KNOWLEDGE_CLAIM_MODEL_PATH,
    'knowledge_nli': KNOWLEDGE_NLI_MODEL_PATH,
    'results_dir': Path("./results"),
}

# ==================== BASELINES (Part A Performance) ====================
STYLE_BASELINE_F1 = 0.92
KNOWLEDGE_BASELINE_F1 = 0.32

# ==================== GRIDSEARCH PARAMETERS ====================

# Stratégie 4: Weighted Voting + Threshold (3 parameters → 125 configs)
WEIGHTS_GRID = {
    'w_style': np.arange(0.4, 0.95, 0.1),      # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    'w_knowledge': np.arange(0.1, 0.6, 0.1),   # [0.1, 0.2, 0.3, 0.4, 0.5]
    'threshold': np.arange(0.3, 0.8, 0.1),     # [0.3, 0.4, 0.5, 0.6, 0.7]
}

# Stratégie 1: Cascading thresholds (1 parameter → ~10 configs)
CASCADE_THRESHOLDS = np.arange(0.5, 0.95, 0.05)

# Stratégie 3: Disagreement weight (1 parameter → ~7 configs)
DISAGREEMENT_WEIGHTS = np.arange(0.3, 0.9, 0.1)

# Stratégie 2: Direct (No gridsearch)
# Stratégie 5: RF meta-learning (No gridsearch parameters, trained on subset)

# ==================== METRICS ====================
METRICS_TO_COMPUTE = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'specificity',
]

# ==================== DEVICE ====================
if 'torch' in locals():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

# ==================== RANDOM STATE ====================
RANDOM_STATE = 42

# ==================== OUTPUT ====================
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Subdirectories
(RESULTS_DIR / "confusion_matrices").mkdir(exist_ok=True)
(RESULTS_DIR / "predictions_per_strategy").mkdir(exist_ok=True)

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
VERBOSE = True

# ==================== PART B SPLIT ====================
PART_B_VALIDATION_SPLIT = 0.5  # 50% validation for tuning, 50% test for eval

print("✅ Configuration fusion RÉVISÉE chargée")
print(f"   - 5 Stratégies à tester")
print(f"   - Stratégie 4: {len(WEIGHTS_GRID['w_style']) * len(WEIGHTS_GRID['w_knowledge']) * len(WEIGHTS_GRID['threshold'])} configurations")
print(f"   - Device: {DEVICE}")

