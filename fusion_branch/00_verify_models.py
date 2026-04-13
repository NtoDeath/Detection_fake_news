"""Phase 6 - Section 1: Verify Frozen Models"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent

print("="*80)
print("Section 1: VÉRIFICATION MODÈLES GELÉS (Part A Training)")
print("="*80)

models_status = {
    'Style best_model': parent_dir / "style_branch" / "results" / "best_model.pkl",
    'Style RoBERTa': parent_dir / "style_branch" / "roberta_fine_tunned",
    'Knowledge Claim Detector': parent_dir / "knowledge_branch" / "claim_detector_model",
    'Knowledge NLI': parent_dir / "knowledge_branch" / "my_claim_model",
}

all_models_ok = True
for model_name, model_path in models_status.items():
    exists = model_path.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {model_name.ljust(30)}: {model_path}")
    if not exists:
        all_models_ok = False

print("="*80)
if all_models_ok:
    print("✅ TOUS LES MODÈLES PRÉSENTS - Prêt pour Phase 6B/6C")
else:
    print("❌ MODÈLES MANQUANTS - Veuillez exécuter Phases 1-5 avant")
    print("   Style Phase: style_branch/")
    print("   Knowledge Phase: knowledge_branch/")
    sys.exit(1)

# Créer dossier résultats
Path("results").mkdir(exist_ok=True)
print("\n✅ Dossier results créé pour stockage résultats")
