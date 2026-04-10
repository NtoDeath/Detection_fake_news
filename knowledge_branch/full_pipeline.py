"""
Pipeline complet de vérification de fake news (Knowledge-based)
- Entrée : un texte/article
- Étapes :
  1. Détection de claims (avec spaCy NER)
  2. Récupération de preuves
  3. Vérification (NLI)
- Sortie : rapport de vérification avec verdicts
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Configuration des paths
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"

if str(KNOWLEDGE_BRANCH) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_BRANCH))

def initialize_pipeline():
    """Initialise tous les composants du pipeline"""
    print("🔧 Initialisation du pipeline complet...")
    
    try:
        from evidence_retrieval import EvidenceRetriever
        from claim_verification import ClaimVerifier
        from transformers import pipeline as hf_pipeline
        import torch
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return None
    
    # Evidence Retriever
    retriever = EvidenceRetriever(
        google_api_key=None,
        google_cse_id="151bf4aa4eae44373",
        wolfram_app_id="LEU7Y6728T",
        config_languages={
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm'
        }
    )
    
    # Claim Verifier
    verifier = ClaimVerifier()
    
    # Claim Detector (optionnel)
    try:
        model_dir = KNOWLEDGE_BRANCH / "my_claim_model"
        if (model_dir / "config.json").exists():
            claim_pipeline = hf_pipeline(
                "text-classification",
                model=str(model_dir),
                device=0 if torch.cuda.is_available() else -1
            )
        else:
            claim_pipeline = None
    except Exception as e:
        print(f"   ⚠️  Claim Detector non disponible : {e}")
        claim_pipeline = None
    
    print("   ✅ Pipeline initialisé\n")
    
    return {
        'retriever': retriever,
        'verifier': verifier,
        'claim_detector': claim_pipeline
    }

def detect_claim_with_ml(text, claim_detector):
    """Détecte si un texte est un claim avec le ML model"""
    if claim_detector is None:
        return None, None
    
    try:
        results = claim_detector(text)
        score = next((r['score'] for r in results if r['label'] == 'LABEL_1'), 0)
        return score > 0.5, score
    except:
        return None, None

def extract_entities(sentence, nlp):
    """Extrait les entités nommées d'une phrase"""
    doc = nlp(sentence)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'PERSON', 'DATE', 'ORG', 'WORK_OF_ART']:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
    return entities

def process_text(text, language='en', pipeline=None):
    """
    Traite un texte complet :
    1. Extraction des phrases
    2. Détection de claims
    3. Récupération de preuves
    4. Vérification
    """
    if pipeline is None:
        print("❌ Pipeline non initialisé")
        return []
    
    retriever = pipeline['retriever']
    verifier = pipeline['verifier']
    claim_detector = pipeline['claim_detector']
    
    nlp = retriever.nlp_models.get(language)
    if nlp is None:
        print(f"❌ Langue '{language}' non supportée")
        return []
    
    doc = nlp(text)
    final_report = []
    
    print(f"\n🚀 --- TRAITEMENT DU TEXTE ({language.upper()}) ---\n")
    
    for sent_idx, sent in enumerate(doc.sents, 1):
        sentence = sent.text.strip()
        if not sentence or len(sentence) < 5:
            continue
        
        print(f"[{sent_idx}] {sentence}")
        
        # ÉTAPE 1 : Détection avec ML (optionnel)
        ml_score = None
        if claim_detector:
            is_claim_ml, ml_score = detect_claim_with_ml(sentence, claim_detector)
            print(f"    ℹ️  ML Score: {ml_score:.3f}" if ml_score else "")
        
        # ÉTAPE 2 : Extraction des entités
        entities = extract_entities(sentence, nlp)
        has_entities = len(entities) > 0
        
        if has_entities:
            entity_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in entities.items()])
            print(f"    🏷️  Entités: {entity_str}")
        
        # Décision : traiter si entités détectées ou si ML le suggère
        should_verify = has_entities
        if ml_score is not None:
            should_verify = should_verify or (ml_score > 0.3)
        
        if not should_verify:
            print(f"    ⏭️  Ignoré\n")
            final_report.append({
                "sentence": sentence,
                "verdict": "NOT_CHECKED",
                "reason": "No entities and low ML score"
            })
            continue
        
        print(f"    🔍 Vérification en cours...")
        
        # ÉTAPE 3 : Récupération de preuves
        evidence = retriever.get_evidence(sentence, language)
        
        if not evidence:
            print(f"    ❌ Aucune preuve trouvée")
            final_report.append({
                "sentence": sentence,
                "verdict": "NO_EVIDENCE",
                "entities": entities,
                "ml_score": ml_score
            })
            print()
            continue
        
        source_name = evidence.get('title', 'Unknown')
        source_url = evidence.get('url', 'N/A')
        
        print(f"    ✅ Source: {source_name}")
        print(f"    📄 Preuve: {evidence['content'][:80]}...")
        
        # ÉTAPE 4 : Vérification
        verdict, confidence = verifier.verify(sentence, evidence['content'])
        
        print(f"    ✅ Verdict: {verdict} (confiance: {confidence:.3f})")
        print()
        
        final_report.append({
            "sentence": sentence,
            "verdict": verdict,
            "confidence": float(confidence),
            "source": source_name,
            "source_url": source_url,
            "entities": entities,
            "ml_score": float(ml_score) if ml_score else None,
            "evidence_snippet": evidence['content'][:150]
        })
    
    return final_report

def save_report(report, output_file=None):
    """Sauvegarde le rapport en JSON"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = KNOWLEDGE_BRANCH / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"verification_report_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return output_file

def display_summary(report):
    """Affiche un résumé du rapport"""
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DU RAPPORT")
    print("=" * 70)
    
    verdicts_count = {}
    for item in report:
        verdict = item.get('verdict', 'UNKNOWN')
        verdicts_count[verdict] = verdicts_count.get(verdict, 0) + 1
    
    print(f"\nTotal d'items traités : {len(report)}")
    print(f"\nVerdicts :")
    for verdict, count in verdicts_count.items():
        percentage = (count / len(report)) * 100 if report else 0
        print(f"  {verdict:20} : {count:3d} ({percentage:5.1f}%)")
    
    print("\nDétails :")
    for i, item in enumerate(report, 1):
        sentence = item.get('sentence', 'N/A')[:60]
        verdict = item.get('verdict', 'UNKNOWN')
        confidence = item.get('confidence', None)
        
        if confidence is not None:
            print(f"  {i}. [{verdict:15}] {sentence}... (conf: {confidence:.3f})")
        else:
            print(f"  {i}. [{verdict:15}] {sentence}...")
    
    print("\n" + "=" * 70)

def main(text=None, language='en'):
    """Fonction principale"""
    print("=" * 70)
    print("🚀 PIPELINE COMPLET - FACT-CHECKING")
    print("=" * 70)
    print()
    
    # Initialiser le pipeline
    pipeline = initialize_pipeline()
    if pipeline is None:
        return
    
    # Texte par défaut ou utilisateur
    if text is None:
        text = """
        Algeria is located in Africa. The Eiffel Tower is in Paris. 
        The Eiffel Tower was built in 1990. Paris is the capital of France.
        Mount Everest is the highest mountain in the world.
        """
    
    print(f"📝 Texte à analyser ({language}):")
    print(f"{text}\n")
    
    # Traiter le texte
    report = process_text(text, language=language, pipeline=pipeline)
    
    # Afficher le résumé
    display_summary(report)
    
    # Sauvegarder le rapport
    output_file = save_report(report)
    print(f"\n💾 Rapport sauvegardé : {output_file}")
    
    return report

def test_multiple_languages():
    """Teste le pipeline sur plusieurs langues"""
    print("=" * 70)
    print("🌍 TEST MULTILINGUE")
    print("=" * 70)
    print()
    
    pipeline = initialize_pipeline()
    if pipeline is None:
        return
    
    test_cases = {
        'en': "The Great Wall of China is one of the largest structures. Paris is in France.",
        'fr': "La France est un pays d'Europe. Paris est la capitale.",
        'es': "La capital de España es Madrid. Barcelona es una ciudad importante."
    }
    
    all_reports = {}
    for language, text in test_cases.items():
        print(f"\n--- {language.upper()} ---")
        report = process_text(text, language=language, pipeline=pipeline)
        all_reports[language] = report
        display_summary(report)
    
    return all_reports

if __name__ == "__main__":
    # Exemple d'utilisation simple
    report = main()
    
    # Décommenter pour tester le multilingue
    # test_multiple_languages()
