"""
Initialisation et Configuration du Pipeline de Récupération et Vérification
- Configuration du Evidence Retriever avec les APIs
- Configuration du Claim Verifier
- Tests simples des composants
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Configuration des paths
PROJECT_ROOT = Path.home() / "Documents/IFT714 Traitement des LN/Projet/Detection_fake_news"
KNOWLEDGE_BRANCH = PROJECT_ROOT / "knowledge_branch"

if str(KNOWLEDGE_BRANCH) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_BRANCH))

def initialize_evidence_retriever():
    """Initialise le Evidence Retriever avec les configurations"""
    print("🔍 Initialisation du Evidence Retriever...")
    
    try:
        from evidence_retrieval import EvidenceRetriever
        print("   ✅ Module evidence_retrieval importé")
    except ImportError as e:
        print(f"   ❌ Erreur d'import : {e}")
        return None
    
    # Configuration des API (à remplacer par vos vraies clés)
    WOLFRAM_APPID = "LEU7Y6728T"
    GOOGLE_API_KEY = None  # À configurer
    GOOGLE_CSE_ID = "151bf4aa4eae44373"
    
    print(f"   Configuration :")
    print(f"      - Wolfram Alpha : {'✅' if WOLFRAM_APPID else '❌'}")
    print(f"      - Google API : {'✅' if GOOGLE_API_KEY else '❌ (optionnel)'}")
    print(f"      - Google CSE : {'✅' if GOOGLE_CSE_ID else '❌'}")
    
    # Initialisation
    retriever = EvidenceRetriever(
        google_api_key=GOOGLE_API_KEY,
        google_cse_id=GOOGLE_CSE_ID,
        wolfram_app_id=WOLFRAM_APPID,
        config_languages={
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm'
        }
    )
    
    print("   ✅ Evidence Retriever initialisé\n")
    return retriever

def initialize_claim_verifier():
    """Initialise le Claim Verifier"""
    print("🔐 Initialisation du Claim Verifier...")
    
    try:
        from claim_verification import ClaimVerifier
        print("   ✅ Module claim_verification importé")
    except ImportError as e:
        print(f"   ❌ Erreur d'import : {e}")
        return None
    
    verifier = ClaimVerifier()
    
    print("   ✅ Claim Verifier initialisé\n")
    return verifier

def test_evidence_retriever(retriever):
    """Teste le retriever avec quelques exemples"""
    print("🧪 Test du Evidence Retriever...")
    
    if retriever is None:
        print("   ⚠️  Retriever non disponible\n")
        return
    
    test_claims = {
        "The Eiffel Tower is 330 meters tall": "en",
        "Le taux de chômage en France est de 7%": "fr",
        "La capital de España es Madrid": "es"
    }
    
    print("   Résultats des recherches :")
    for claim, language in test_claims.items():
        evidence = retriever.get_evidence(claim, language)
        
        if evidence:
            source_name = evidence.get('title', 'Inconnue')
            print(f"   ✅ [{source_name}] ({language})")
            print(f"      Phrase : {claim}")
            print(f"      Preuve : {evidence['content'][:100]}...")
            print(f"      URL : {evidence.get('url', 'N/A')}")
        else:
            print(f"   ❌ Aucune preuve trouvée : {claim} ({language})")
    print()

def test_claim_verifier(verifier):
    """Teste le verifier avec un exemple simple"""
    print("🧪 Test du Claim Verifier...")
    
    if verifier is None:
        print("   ⚠️  Verifier non disponible\n")
        return
    
    claim = "New York is in the United States"
    evidence = "New York is one of the biggest cities of the United States"
    
    result = verifier.verify(claim, evidence)
    
    print(f"   Claim : {claim}")
    print(f"   Evidence : {evidence}")
    print(f"   Verdict : {result}\n")

def test_full_pipeline(retriever, verifier, claim_detector=None):
    """Teste le pipeline complet : détection -> récupération -> vérification"""
    print("🧪 Test du Pipeline Complet...")
    
    if retriever is None or verifier is None:
        print("   ⚠️  Pipeline incomplet\n")
        return
    
    text_to_check = "Tom Cruise was born in 1962"
    
    print(f"   Claim à tester : {text_to_check}")
    print()
    
    # Optionallement : test avec claim detector
    if claim_detector:
        print("   [STEP 1] Détection du claim...")
        from transformers import pipeline
        try:
            claim_pipeline = pipeline(
                "text-classification",
                model=str(KNOWLEDGE_BRANCH / "my_claim_model"),
                device=-1
            )
            results = claim_pipeline(text_to_check)
            score = next((r['score'] for r in results if r['label'] == 'LABEL_1'), 0)
            print(f"   ✅ Score de détection : {score:.3f}\n")
        except Exception as e:
            print(f"   ⚠️  Claim detector non disponible : {e}\n")
    
    # Étape 2 : Récupération
    print("   [STEP 2] Récupération de preuves...")
    evidence = retriever.get_evidence(text_to_check, language='en')
    
    if evidence:
        print(f"   ✅ Source trouvée : {evidence.get('title', 'Unknown')}")
        print(f"      Contenu : {evidence['content'][:100]}...\n")
        
        # Étape 3 : Vérification
        print("   [STEP 3] Vérification...")
        verdict, confidence = verifier.verify(text_to_check, evidence['content'])
        print(f"   ✅ Verdict : {verdict}")
        print(f"      Confiance : {confidence:.3f}")
        print(f"      URL : {evidence.get('url', 'N/A')}\n")
    else:
        print(f"   ❌ Aucune preuve trouvée\n")

def save_configuration():
    """Sauvegarde la configuration pour utilisation ultérieure"""
    print("💾 Sauvegarde de la configuration...")
    
    config = {
        'WOLFRAM_APPID': 'LEU7Y6728T',
        'GOOGLE_API_KEY': None,
        'GOOGLE_CSE_ID': '151bf4aa4eae44373',
        'languages': {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm'
        }
    }
    
    # Vous pouvez ajouter la sauvegarde JSON ici si nécessaire
    print("   ✅ Configuration définie\n")
    
    return config

def main():
    """Fonction principale : orchestre l'initialisation et les tests"""
    print("=" * 60)
    print("🚀 INITIALIZE KNOWLEDGE BRANCH")
    print("=" * 60)
    print()
    
    # Initialiser les composants
    retriever = initialize_evidence_retriever()
    verifier = initialize_claim_verifier()
    
    # Sauvegarder la configuration
    config = save_configuration()
    
    # Effectuer les tests
    test_evidence_retriever(retriever)
    test_claim_verifier(verifier)
    test_full_pipeline(retriever, verifier, claim_detector=True)
    
    print("=" * 60)
    print("✅ INITIALIZATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
