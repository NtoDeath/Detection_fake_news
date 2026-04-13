"""Phase 6 - Section 6: Strategy 3 - Disagreement-Adaptive Weighting"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from strategy_3_disagreement_adaptive import DisagreementAdaptiveWeighting

print("\n" + "="*80)
print("Section 6: STRATÉGIE 3 - DISAGREEMENT-ADAPTIVE WEIGHTING")
print("="*80)

results_dir = script_dir / "results"

# Charger données
with open(results_dir / "part_b_split.pkl", 'rb') as f:
    split_data = pickle.load(f)

style_preds_train = split_data['style_preds_train']
style_confs_train = split_data['style_confs_train']
knowledge_preds_train = split_data['knowledge_preds_train']
knowledge_confs_train = split_data['knowledge_confs_train']
y_train = split_data['y_train']
style_preds_test = split_data['style_preds_test']
style_confs_test = split_data['style_confs_test']
knowledge_preds_test = split_data['knowledge_preds_test']
knowledge_confs_test = split_data['knowledge_confs_test']
y_test = split_data['y_test']

# Instancier
disagree_adaptive = DisagreementAdaptiveWeighting()
best_disagree_weight = None
best_disagree_f1 = 0

print("\n🔍 Gridsearch sur fusion_train...")
for weight in np.arange(0.3, 0.85, 0.1):
    disagree_adaptive.disagreement_weight = weight
    preds_train, _ = disagree_adaptive.predict(
        style_preds_train, style_confs_train,
        knowledge_preds_train, knowledge_confs_train
    )
    f1 = f1_score(y_train, preds_train)
    
    if f1 > best_disagree_f1:
        best_disagree_f1 = f1
        best_disagree_weight = weight

print(f"✅ Meilleur poids trouvé: {best_disagree_weight:.2f} (F1 train: {best_disagree_f1:.4f})")

# Évaluer sur fusion_test
disagree_adaptive.disagreement_weight = best_disagree_weight
preds_disagree, _ = disagree_adaptive.predict(
    style_preds_test, style_confs_test,
    knowledge_preds_test, knowledge_confs_test
)

da_acc = accuracy_score(y_test, preds_disagree)
da_prec = precision_score(y_test, preds_disagree, zero_division=0)
da_rec = recall_score(y_test, preds_disagree, zero_division=0)
da_f1 = f1_score(y_test, preds_disagree, zero_division=0)

print(f"\n📊 Résultats Stratégie 3 (Test unseen):")
print(f"   Accuracy:  {da_acc:.4f}")
print(f"   Precision: {da_prec:.4f}")
print(f"   Recall:    {da_rec:.4f}")
print(f"   F1-Score:  {da_f1:.4f}")

# Sauvegarder résultats
da_report = {
    'strategy_name': 'Disagreement-Adaptive Weighting',
    'best_disagreement_weight': float(best_disagree_weight),
    'accuracy': float(da_acc),
    'precision': float(da_prec),
    'recall': float(da_rec),
    'f1_score': float(da_f1)
}
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "strategy_3_disagreement_report.json", 'w') as f:
    json.dump(da_report, f, indent=2)

print(f"✅ Résultats sauvegardés: strategy_3_disagreement_report.json")
print("="*80)
