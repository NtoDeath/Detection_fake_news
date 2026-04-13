"""Phase 6 - Section 3: Split Part B 50/50"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

script_dir = Path(__file__).parent.absolute()

print("\n" + "="*80)
print("Section 3: SPLIT PART B EN FUSION_TRAIN/TEST (50/50)")
print("="*80)

# Charger prédictions
predictions_file = script_dir / "results" / "part_b_predictions.pkl"
with open(predictions_file, 'rb') as f:
    predictions_data = pickle.load(f)

part_B = predictions_data['part_B']
style_preds = predictions_data['style_predictions']
knowledge_preds = predictions_data['knowledge_predictions']
y_true = part_B['label'].values

# Utiliser les prédictions comme confidences (déjà en [0,1])
style_confs = style_preds
knowledge_confs = knowledge_preds

# Créer binary predictions en thresholdant les confidences à 0.5
style_preds_binary = (style_preds >= 0.5).astype(int)
knowledge_preds_binary = (knowledge_preds >= 0.5).astype(int)

# Stratified split 50/50
indices = np.arange(len(y_true))
train_idx, test_idx = train_test_split(
    indices, 
    test_size=0.5, 
    stratify=y_true, 
    random_state=42
)

y_train, y_test = y_true[train_idx], y_true[test_idx]

print(f"\n📊 Split Results:")
print(f"   Fusion Train: {len(y_train)} samples (Classe 0: {(y_train==0).sum()}, Classe 1: {(y_train==1).sum()})")
print(f"   Fusion Test:  {len(y_test)} samples (Classe 0: {(y_test==0).sum()}, Classe 1: {(y_test==1).sum()})")

# Créer DataFrames pour sauvegarde
df_train = pd.DataFrame({
    'style_pred': style_preds[train_idx],
    'style_conf': style_confs[train_idx],
    'knowledge_pred': knowledge_preds[train_idx],
    'knowledge_conf': knowledge_confs[train_idx],
    'label_true': y_train
})

df_test = pd.DataFrame({
    'style_pred': style_preds[test_idx],
    'style_conf': style_confs[test_idx],
    'knowledge_pred': knowledge_preds[test_idx],
    'knowledge_conf': knowledge_confs[test_idx],
    'label_true': y_test
})

results_dir = script_dir / "results"
results_dir.mkdir(exist_ok=True)

df_train.to_csv(results_dir / "fusion_train.csv", index=False)
df_test.to_csv(results_dir / "fusion_test.csv", index=False)

# Sauvegarder aussi les arrays pour utilisation dans stratégies
split_data = {
    'style_preds_train': style_preds_binary[train_idx],
    'style_confs_train': style_confs[train_idx],
    'knowledge_preds_train': knowledge_preds_binary[train_idx],
    'knowledge_confs_train': knowledge_confs[train_idx],
    'y_train': y_train,
    'style_preds_test': style_preds_binary[test_idx],
    'style_confs_test': style_confs[test_idx],
    'knowledge_preds_test': knowledge_preds_binary[test_idx],
    'knowledge_confs_test': knowledge_confs[test_idx],
    'y_test': y_test
}

with open(results_dir / "part_b_split.pkl", 'wb') as f:
    pickle.dump(split_data, f)

print(f"\n✅ Données de fusion sauvegardées:")
print(f"   - fusion_train.csv: {len(df_train)} rows")
print(f"   - fusion_test.csv: {len(df_test)} rows")
print(f"   - results/part_b_split.pkl: données splits pour stratégies")
print("="*80)
