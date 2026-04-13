"""Phase 6 - Section 2: Load Part B & Generate Raw Predictions"""

import sys
import subprocess
from pathlib import Path
import pickle

script_dir = Path(__file__).parent.absolute()

print("\n" + "="*80)
print("Section 2: CHARGER PART B ET GÉNÉRER PRÉDICTIONS BRUTES")
print("="*80)

# Exécuter evidence_loader pour générer les prédictions
import importlib.util
evidence_loader_path = script_dir / "evidence_loader.py"
spec = importlib.util.spec_from_file_location("evidence_loader", str(evidence_loader_path))
evidence_loader = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(evidence_loader)
    print("✅ evidence_loader exécuté avec succès")
except Exception as e:
    print(f"❌ Error executing evidence_loader: {e}")
    import traceback
    traceback.print_exc()

# Charger les prédictions générées
predictions_file = script_dir / "results" / "part_b_predictions.pkl"
if predictions_file.exists():
    with open(predictions_file, 'rb') as f:
        predictions_data = pickle.load(f)
    
    part_B = predictions_data['part_B']
    y_true = part_B['label'].values
    
    print(f"\n✅ Prédictions chargées:")
    print(f"   - Style predictions: {len(predictions_data['style_predictions'])} samples")
    print(f"   - Knowledge predictions: {len(predictions_data['knowledge_predictions'])} samples")
    print(f"   - Ground truth labels: {len(y_true)} samples")
    print(f"   - Classe 0 (FAKE): {(y_true==0).sum()} ({(y_true==0).sum()/len(y_true)*100:.1f}%)")
    print(f"   - Classe 1 (TRUE): {(y_true==1).sum()} ({(y_true==1).sum()/len(y_true)*100:.1f}%)")
else:
    print("❌ Fichier predictions non trouvé - evidence_loader.py a échoué")
    sys.exit(1)

print("="*80)
