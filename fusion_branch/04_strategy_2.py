"""Phase 6 - Section 5: Strategy 2 - Confidence-Weighted Voting"""

import sys
import pickle
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from strategy_2_confidence_weighted import ConfidenceWeightedVoting

print("\n" + "="*80)
print("Section 5: STRATÉGIE 2 - CONFIDENCE-WEIGHTED VOTING")
print("="*80)

results_dir = script_dir / "results"

# Charger données
with open(results_dir / "part_b_split.pkl", 'rb') as f:
    split_data = pickle.load(f)

style_preds_test = split_data['style_preds_test']
style_confs_test = split_data['style_confs_test']
knowledge_preds_test = split_data['knowledge_preds_test']
knowledge_confs_test = split_data['knowledge_confs_test']
y_test = split_data['y_test']

# Instancier (pas de paramètres)
cf_voting = ConfidenceWeightedVoting()

# Évaluer directement sur fusion_test
preds_cf, _ = cf_voting.predict(style_preds_test, style_confs_test,
                                 knowledge_preds_test, knowledge_confs_test)

cf_acc = accuracy_score(y_test, preds_cf)
cf_prec = precision_score(y_test, preds_cf, zero_division=0)
cf_rec = recall_score(y_test, preds_cf, zero_division=0)
cf_f1 = f1_score(y_test, preds_cf, zero_division=0)

print(f"\n📊 Résultats Stratégie 2 (Test unseen):")
print(f"   Accuracy:  {cf_acc:.4f}")
print(f"   Precision: {cf_prec:.4f}")
print(f"   Recall:    {cf_rec:.4f}")
print(f"   F1-Score:  {cf_f1:.4f}")
print(f"   (Note: pas de gridsearch - poids fixes 0.92/0.32)")

# Sauvegarder résultats
cf_report = {
    'strategy_name': 'Confidence-Weighted Voting',
    'w_style': 0.92,
    'w_knowledge': 0.32,
    'accuracy': float(cf_acc),
    'precision': float(cf_prec),
    'recall': float(cf_rec),
    'f1_score': float(cf_f1)
}
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "strategy_2_cf_voting_report.json", 'w') as f:
    json.dump(cf_report, f, indent=2)

print(f"✅ Résultats sauvegardés: strategy_2_cf_voting_report.json")
print("="*80)
