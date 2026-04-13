"""Phase 6 - Section 4: Strategy 1 - Cascading (Style First)"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Import stratégie
from strategy_1_cascading import CascadingStyleFirst

print("\n" + "="*80)
print("Section 4: STRATÉGIE 1 - CASCADING (Style First)")
print("="*80)

results_dir = script_dir / "results"

# Charger données splittées
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

# Instancier et optimiser
cascading = CascadingStyleFirst()
best_cascading_threshold = None
best_cascading_f1 = 0

print("\n🔍 Gridsearch sur fusion_train...")
for threshold in np.arange(0.5, 1.0, 0.05):
    cascading.style_threshold = threshold
    preds_train, _ = cascading.predict(style_preds_train, style_confs_train, 
                                        knowledge_preds_train, knowledge_confs_train)
    f1 = f1_score(y_train, preds_train)
    
    if f1 > best_cascading_f1:
        best_cascading_f1 = f1
        best_cascading_threshold = threshold

print(f"✅ Meilleur seuil trouvé: {best_cascading_threshold:.2f} (F1 train: {best_cascading_f1:.4f})")

# Évaluer sur fusion_test
cascading.style_threshold = best_cascading_threshold
preds_test, _ = cascading.predict(style_preds_test, style_confs_test,
                                   knowledge_preds_test, knowledge_confs_test)

cascading_acc = accuracy_score(y_test, preds_test)
cascading_prec = precision_score(y_test, preds_test, zero_division=0)
cascading_rec = recall_score(y_test, preds_test, zero_division=0)
cascading_f1 = f1_score(y_test, preds_test, zero_division=0)

print(f"\n📊 Résultats Stratégie 1 (Test unseen):")
print(f"   Accuracy:  {cascading_acc:.4f}")
print(f"   Precision: {cascading_prec:.4f}")
print(f"   Recall:    {cascading_rec:.4f}")
print(f"   F1-Score:  {cascading_f1:.4f}")

# Sauvegarder résultats
cascading_report = {
    'strategy_name': 'Cascading Style First',
    'best_threshold': float(best_cascading_threshold),
    'accuracy': float(cascading_acc),
    'precision': float(cascading_prec),
    'recall': float(cascading_rec),
    'f1_score': float(cascading_f1)
}
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "strategy_1_cascading_report.json", 'w') as f:
    json.dump(cascading_report, f, indent=2)

print(f"✅ Résultats sauvegardés: strategy_1_cascading_report.json")
print("="*80)
