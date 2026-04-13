"""Phase 6 - Section 7: Strategy 4 - Weighted + Threshold (Gridsearch)"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from strategy_4_weighted_threshold import WeightedVotingWithThreshold

print("\n" + "="*80)
print("Section 7: STRATÉGIE 4 - WEIGHTED + THRESHOLD ⭐")
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
weighted_threshold = WeightedVotingWithThreshold()

# Gridsearch sur les 3 paramètres
best_w_style = None
best_w_knowledge = None
best_threshold = None
best_wt_f1 = 0

print("\n🔍 Gridsearch sur fusion_train (125 configurations)...")
total_configs = 0
for w_s in np.arange(0.5, 1.0, 0.1):
    for w_k in np.arange(0.05, 0.55, 0.1):
        for thresh in np.arange(0.4, 0.65, 0.05):
            total_configs += 1
            weighted_threshold.w_style = w_s
            weighted_threshold.w_knowledge = w_k
            weighted_threshold.threshold = thresh
            preds_train, _ = weighted_threshold.predict(
                style_preds_train, style_confs_train,
                knowledge_preds_train, knowledge_confs_train
            )
            f1 = f1_score(y_train, preds_train)
            
            if f1 > best_wt_f1:
                best_wt_f1 = f1
                best_w_style = w_s
                best_w_knowledge = w_k
                best_threshold = thresh

print(f"✅ Configurations testées: {total_configs}")
print(f"✅ Meilleurs paramètres trouvés:")
print(f"   w_style={best_w_style:.2f}, w_knowledge={best_w_knowledge:.2f}, threshold={best_threshold:.2f}")
print(f"   F1 train: {best_wt_f1:.4f}")

# Évaluer sur fusion_test
weighted_threshold.w_style = best_w_style
weighted_threshold.w_knowledge = best_w_knowledge
weighted_threshold.threshold = best_threshold
preds_wt, _ = weighted_threshold.predict(
    style_preds_test, style_confs_test,
    knowledge_preds_test, knowledge_confs_test
)

wt_acc = accuracy_score(y_test, preds_wt)
wt_prec = precision_score(y_test, preds_wt, zero_division=0)
wt_rec = recall_score(y_test, preds_wt, zero_division=0)
wt_f1 = f1_score(y_test, preds_wt, zero_division=0)

print(f"\n📊 Résultats Stratégie 4 (Test unseen):")
print(f"   Accuracy:  {wt_acc:.4f}")
print(f"   Precision: {wt_prec:.4f}")
print(f"   Recall:    {wt_rec:.4f}")
print(f"   F1-Score:  {wt_f1:.4f}")

# Sauvegarder résultats
wt_report = {
    'strategy_name': 'Weighted Voting + Threshold',
    'best_params': {
        'w_style': float(best_w_style),
        'w_knowledge': float(best_w_knowledge),
        'threshold': float(best_threshold)
    },
    'gridsearch_configs_tested': total_configs,
    'accuracy': float(wt_acc),
    'precision': float(wt_prec),
    'recall': float(wt_rec),
    'f1_score': float(wt_f1)
}
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "strategy_4_weighted_threshold_report.json", 'w') as f:
    json.dump(wt_report, f, indent=2)

print(f"✅ Résultats sauvegardés: strategy_4_weighted_threshold_report.json")
print("="*80)
