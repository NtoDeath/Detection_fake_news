"""Phase 6 - Section 8: Strategy 5 - Stacked RandomForest"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from strategy_5_stacked_rf import StackedRandomForestFusion

print("\n" + "="*80)
print("Section 8: STRATÉGIE 5 - STACKED RANDOMFOREST ⭐")
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

# Instancier et entraîner
stacked_rf = StackedRandomForestFusion()

print("\n🔧 Entraînement du meta-learner RandomForest...")
stacked_rf.train(
    style_preds_train, style_confs_train,
    knowledge_preds_train, knowledge_confs_train,
    y_train
)
print("✅ Entraînement complété")

# Évaluer sur fusion_test
preds_rf, _ = stacked_rf.predict(
    style_preds_test, style_confs_test,
    knowledge_preds_test, knowledge_confs_test
)

rf_acc = accuracy_score(y_test, preds_rf)
rf_prec = precision_score(y_test, preds_rf, zero_division=0)
rf_rec = recall_score(y_test, preds_rf, zero_division=0)
rf_f1 = f1_score(y_test, preds_rf, zero_division=0)

print(f"\n📊 Résultats Stratégie 5 (Test unseen):")
print(f"   Accuracy:  {rf_acc:.4f}")
print(f"   Precision: {rf_prec:.4f}")
print(f"   Recall:    {rf_rec:.4f}")
print(f"   F1-Score:  {rf_f1:.4f}")

# Afficher feature importance
importances = None
if hasattr(stacked_rf.meta_model, 'feature_importances_'):
    importances = stacked_rf.meta_model.feature_importances_
    print(f"\n🔍 Feature Importance (Meta-learner):")
    print(f"   style_pred:     {importances[0]:.4f}")
    print(f"   style_conf:     {importances[1]:.4f}")
    print(f"   knowledge_pred: {importances[2]:.4f}")
    print(f"   knowledge_conf: {importances[3]:.4f}")

# Sauvegarder résultats
rf_report = {
    'strategy_name': 'Stacked RandomForest',
    'model_type': 'RandomForest (n_estimators=100, max_depth=5)',
    'training_samples': len(y_train),
    'accuracy': float(rf_acc),
    'precision': float(rf_prec),
    'recall': float(rf_rec),
    'f1_score': float(rf_f1),
    'feature_importance': {
        'style_pred': float(importances[0]) if importances is not None else None,
        'style_conf': float(importances[1]) if importances is not None else None,
        'knowledge_pred': float(importances[2]) if importances is not None else None,
        'knowledge_conf': float(importances[3]) if importances is not None else None
    }
}
results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)
with open(results_dir / "strategy_5_stacked_rf_report.json", 'w') as f:
    json.dump(rf_report, f, indent=2)

print(f"✅ Résultats sauvegardés: strategy_5_stacked_rf_report.json")
print("="*80)
