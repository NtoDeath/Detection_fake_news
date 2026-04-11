"""
Évaluation complète du Pipeline Knowledge-based (80% Part A)
- Chargement du dataset FEVER train_partA.jsonl
- Évaluation de la récupération de preuves
- Évaluation de la vérification des claims
- Génération des métriques et rapports
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des paths
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"
DATA_DIR = PROJECT_ROOT / "data" / "knowledge_based"

if str(KNOWLEDGE_BRANCH) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_BRANCH))

def load_fever_dataset():
    """Charge et prépare le dataset FEVER (80% Part A)"""
    print("📂 Chargement du dataset FEVER train_partA.jsonl (80% Part A)...")
    
    fever_file = KNOWLEDGE_BRANCH / 'splits' / 'train_partA.jsonl'
    
    if not fever_file.exists():
        print(f"   ❌ Fichier non trouvé : {fever_file}")
        print("   Exécutez d'abord : python prepare_part_B_heterogeneous.py")
        return None
    
    df_fever = pd.read_json(str(fever_file), lines=True)
    
    # Renommer les labels pour cohérence
    mapping = {
        'SUPPORTS': 'SUPPORTED',
        'REFUTES': 'REFUTED',
        'NOT ENOUGH INFO': 'NEUTRAL / NOT ENOUGH INFO'
    }
    df_fever['label'] = df_fever['label'].replace(mapping)
    
    print(f"   📊 Total : {len(df_fever)} instances")
    print(f"   📈 Distribution des labels :")
    print(df_fever['label'].value_counts())
    
    return df_fever

def balance_dataset(df_fever, n_per_class=30):
    """Équilibre le dataset pour l'évaluation"""
    print(f"\n⚖️  Équilibrage du dataset ({n_per_class} par classe)...")
    
    balanced_dfs = []
    for label_value in df_fever['label'].unique():
        subset = df_fever[df_fever['label'] == label_value].sample(
            n=min(n_per_class, len(df_fever[df_fever['label'] == label_value])),
            random_state=42
        )
        balanced_dfs.append(subset)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Mélanger le dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_data_list = balanced_df[['claim', 'label']].to_dict('records')
    
    print(f"   ✅ Dataset équilibré : {len(test_data_list)} instances")
    print(f"   📈 Distribution :")
    print(balanced_df['label'].value_counts())
    
    return test_data_list

def evaluate_retrieval(retriever, test_set):
    """Évalue la capacité du retriever à récupérer des preuves"""
    print("\n🔍 Évaluation du Evidence Retriever...")
    print("   Analyse de récupération par classe :")
    
    stats = {}
    for item in test_set:
        label = item['label']
        if label not in stats:
            stats[label] = {'total': 0, 'retrieved': 0}
        stats[label]['total'] += 1
        
        evidence = retriever.get_evidence(item['claim'])
        
        if evidence and evidence.get('content') and len(evidence['content']) > 20:
            stats[label]['retrieved'] += 1
    
    print()
    for label, count in stats.items():
        rate = (count['retrieved'] / count['total']) * 100 if count['total'] > 0 else 0
        print(f"   {label:25} : {rate:5.1f}% ({count['retrieved']:2d}/{count['total']:2d})")
    
    return stats

def evaluate_verification(verifier, retriever, test_set):
    """Évalue la vérification des claims"""
    print("\n🔐 Évaluation du Claim Verifier...")
    
    y_true = []
    y_pred = []
    retrieval_success = 0
    
    print("   Traitement des claims...")
    
    for i, item in enumerate(test_set):
        claim = item['claim']
        true_label = item['label']
        
        # Récupération
        evidence = retriever.get_evidence(claim, language='en')
        
        if evidence and evidence.get('content'):
            retrieval_success += 1
            evidence_text = evidence['content']
        else:
            evidence_text = ""
        
        # Vérification
        pred_label, score = verifier.verify(claim, evidence_text)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        if (i + 1) % 10 == 0:
            print(f"      {i + 1}/{len(test_set)} claims traités...")
    
    # Calcul des métriques
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS DE L'ÉVALUATION (80% Part A)")
    print("=" * 50)
    
    print(f"\n📈 Taux de récupération : {retrieval_success}/{len(test_set)} ({100*retrieval_success/len(test_set):.1f}%)")
    
    print("\n📋 Rapport de Classification :")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Matrice de confusion
    labels_order = ["SUPPORTED", "REFUTED", "NEUTRAL / NOT ENOUGH INFO"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    
    return y_true, y_pred, cm, labels_order

def plot_confusion_matrix(cm, labels):
    """Affiche et sauvegarde la matrice de confusion"""
    print("📊 Génération de la matrice de confusion...")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["SUPP", "REFUT", "NEUT"],
        yticklabels=["SUPP", "REFUT", "NEUT"],
        cbar_kws={'label': 'Nombre d\'instances'}
    )
    plt.ylabel('Vrai Label')
    plt.xlabel('Prédiction')
    plt.title('Matrice de Confusion - Pipeline Knowledge-based (80% Part A)')
    plt.tight_layout()
    
    # Sauvegarder
    output_file = KNOWLEDGE_BRANCH / "results" / "confusion_matrix_partA.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Sauvegardée : {output_file}")
    plt.close()

def generate_report(y_true, y_pred, cm, labels_order, retrieval_stats):
    """Génère un rapport d'évaluation complet"""
    print("\n📝 Génération du rapport d'évaluation...")
    
    report_file = KNOWLEDGE_BRANCH / "results" / "evaluation_report_partA.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPPORT D'ÉVALUATION - PIPELINE KNOWLEDGE-BASED (80% Part A)\n")
        f.write("=" * 70 + "\n\n")
        
        # Résumé général
        f.write("RÉSUMÉ GÉNÉRAL\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nombre total d'instances : {len(y_true)}\n")
        f.write(f"Accuracy globale : {accuracy_score(y_true, y_pred):.4f}\n\n")
        
        # Métriques par classe
        f.write("MÉTRIQUES PAR CLASSE\n")
        f.write("-" * 70 + "\n")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        
        for i, label in enumerate(set(y_true)):
            f.write(f"\n{label}:\n")
            f.write(f"  Precision : {precision[i]:.4f}\n")
            f.write(f"  Recall    : {recall[i]:.4f}\n")
            f.write(f"  F1-score  : {f1[i]:.4f}\n")
            f.write(f"  Support   : {support[i]}\n")
        
        # Statistiques de récupération
        f.write("\n\nSTATISTIQUES DE RÉCUPÉRATION\n")
        f.write("-" * 70 + "\n")
        for label, stats in retrieval_stats.items():
            rate = (stats['retrieved'] / stats['total']) * 100 if stats['total'] > 0 else 0
            f.write(f"{label:25} : {rate:5.1f}% ({stats['retrieved']}/{stats['total']})\n")
        
        # Matrice de confusion
        f.write("\n\nMATRICE DE CONFUSION\n")
        f.write("-" * 70 + "\n")
        f.write(f"         {'SUPP':>8} {'REFUT':>8} {'NEUT':>8}\n")
        f.write("-" * 35 + "\n")
        for i, label in enumerate(["SUPP", "REFUT", "NEUT"]):
            f.write(f"{label:8} {cm[i,0]:8d} {cm[i,1]:8d} {cm[i,2]:8d}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"   ✅ Rapport sauvegardé : {report_file}")
    
    # Afficher le contenu
    with open(report_file, 'r', encoding='utf-8') as f:
        print("\n" + f.read())

def main():
    """Fonction principale : orchestre l'évaluation complète"""
    print("=" * 70)
    print("🚀 EVALUATION - KNOWLEDGE BRANCH PIPELINE (80% Part A)")
    print("=" * 70)
    print()
    
    # Importer les modules
    try:
        from evidence_retrieval import EvidenceRetriever
        from claim_verification import ClaimVerifier
        print("✅ Modules importés\n")
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return
    
    # Charger le dataset FEVER
    df_fever = load_fever_dataset()
    if df_fever is None:
        return
    
    # Équilibrer le dataset
    test_set = balance_dataset(df_fever, n_per_class=30)
    
    # Initialiser les composants
    print("\n🔧 Initialisation des composants...")
    retriever = EvidenceRetriever(
        google_api_key=None,
        google_cse_id="151bf4aa4eae44373",
        wolfram_app_id="LEU7Y6728T"
    )
    verifier = ClaimVerifier()
    print("✅ Composants prêts\n")
    
    # Évaluation
    retrieval_stats = evaluate_retrieval(retriever, test_set)
    y_true, y_pred, cm, labels_order = evaluate_verification(verifier, retriever, test_set)
    
    # Visualisation et rapport
    plot_confusion_matrix(cm, labels_order)
    generate_report(y_true, y_pred, cm, labels_order, retrieval_stats)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
