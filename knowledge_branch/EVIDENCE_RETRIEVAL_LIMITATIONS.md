# Limitations & Améliorations du Evidence Retrieval

## 📊 Problème actuel: Limite de 1200 caractères

### Le contexte
Dans `evidence_retrieval.py`, on récupère seulement les **1200 premiers caractères** du contenu Wikipedia:

```python
content = page.content[:1200]  # ← Limite trop restrictive
```

### Pourquoi c'est un problème

#### 1. Sections importantes peuvent être coupées
- Le premier paragraphe peut faire 500-800 chars
- Les définitions/contexte essentiels peuvent être dans les 2e-3e paragraphes
- Les chiffres, faits clés peuvent être après la limite

**Exemple - "Moon landing conspiracy theories":**
```
Paragraphe 1 (200 chars): Description générale de la théorie
Paragraphe 2 (300 chars): Arguments des conspirationistes
Paragraphe 3 (400 chars): ← REFUTATIONS SCIENTIFIQUES (COUPÉES!)
```

#### 2. DeBERTa NLI devient aveugle
Le modèle `claim_verification.py` reçoit:
```
"Moon landing conspiracy theories [SEP] The moon landing was faked..."
```

Au lieu de:
```
"The Moon landing was real. NASA missions confirmed it. 
Independent Chinese and Indian probes photographed landing sites. [SEP] 
The moon landing was faked..."
```

Résultat: **NOT_ENOUGH_INFO** (correct techniquement, mais inutile)

---

## 🔧 Améliorations possibles

### 1. Augmenter la taille du contenu
```python
# Actuel (trop petit)
content = page.content[:1200]

# Proposé (plus substantif)
content = page.content[:3000]  # ou même 5000

# Mais risque: trop volumineux → token limit du modèle NLI
```

**Risque:** DeBERTa a un token limit (~512 tokens)

---

### 2. Filtrer les pages non-substantives
```python
# Détecter les pages vides/stub
if len(page.content.strip()) < 300:
    return None  # Page vide = pas d'évidence utile
```

**Bénéfice:** Évite les faux NOT_ENOUGH_INFO sur des pages de disambiguation

---

### 3. Extraire des sections pertinentes
Au lieu de prendre linéairement le début:

```python
def extract_relevant_sections(page, claim):
    """Extraire les sections les plus pertinentes"""
    sections = page.sections  # Si disponible via API Wikipedia
    
    # Lancer une recherche de similarité par section
    # pour trouver les sections les plus proches du claim
    # Exemple: chercher "moon landing", "conspiracy" en priorité
```

**Bénéfice:** Cibler le contenu pertinent plutôt que juste le début

---

### 4. Combiner plusieurs sources + validation
```python
def get_evidence_with_confidence(claim):
    """Récupérer évidence avec score de qualité"""
    
    sources = []
    
    # Source 1: Wikipedia
    wiki_ev = get_wikipedia_evidence(claim)
    if wiki_ev and len(wiki_ev['content']) > 500:  # Validation min
        sources.append(('Wikipedia', wiki_ev, confidence=0.8))
    
    # Source 2: Wolfram (plus fiable pour faits objectifs)
    wolfram_ev = get_wolfram_evidence(claim)
    if wolfram_ev:
        sources.append(('WolframAlpha', wolfram_ev, confidence=0.95))
    
    # Source 3: Google (sites spécialisés fact-checking)
    google_ev = get_google_evidence(claim)
    if google_ev:
        sources.append(('Google', google_ev, confidence=0.6))
    
    # Retourner la meilleure source + faire la vérification
    if sources:
        best_source = max(sources, key=lambda x: x[2])
        return best_source
```

**Bénéfice:** Score de confiance par source → choix intelligent

---

### 5. Pipeline d'amélioration proposé

```
Claim: "The moon landing was faked"
   ↓
[STAGE 1] Wolfram Alpha Query
├─ "moon landing real"
├─ Si réponse > 200 chars → Utiliser directement (confidence: 95%)
└─ Sinon → continuer
   ↓
[STAGE 2] Wikipedia Search
├─ Chercher "Apollo 11 mission"
├─ Récupérer 3000-5000 chars (pas 1200)
├─ Valider: contenu substantif? (>500 chars)
└─ Si vide → continuer
   ↓
[STAGE 3] Extraction d'entités + Recherche raffinée
├─ NER: extraire "Neil Armstrong", "1969", "NASA"
├─ Re-chercher Wikipedia avec entités
├─ Agglomérer résumés de plusieurs articles
└─ Si rien → continuer
   ↓
[STAGE 4] Google Fact-Check Sites
├─ Chercher sur snopes.com, politifact.com
├─ Récupérer snippet (généralement des vrais fact-checks)
└─ Confiance: 60-80%
   ↓
[RÉSULTAT] Evidence + Confidence Score
├─ Si trouvé: DeBERTa NLI (SUPPORTED/REFUTED/NEUTRAL)
└─ Si rien trouvé: Honnête "NOT_ENOUGH_INFO" (confiance: 99%)
```

---

## 📊 Impact estimé

### Avant (actuellement):
```
"Moon landing faked" 
→ Wikipedia retourne 1200 chars de intro vide
→ DeBERTa recoit "Moon landing conspiracy theories"
→ NOT_ENOUGH_INFO (98.3% confiance) ← INCORRECT
```

### Après (avec améliorations):
```
"Moon landing faked"
→ Wolfram: "Apollo 11 landed July 21, 1969. NASA confirmed..."
→ DeBERTa recoit contenu substantif
→ REFUTED (ou SUPPORTED) (95% confiance) ← CORRECT
```

### Amélioration attendue:
- ✅ Réduction des "NOT_ENOUGH_INFO" faux
- ✅ Augmentation de la confiance dans les vrais verdicts
- ✅ Meilleure couverture de domaines (sciences, histoire, célébrités)

---

## ⚠️ Défis à résoudre

### 1. Token Limit du modèle
- DeBERTa-base: ~512 tokens max
- Wikipedia full content: peut être 5000+ chars = 1000+ tokens
- **Solution:** Résumer/tronquer intelligemment avant NLI

### 2. Latence des APIs
- Wolfram Alpha: 1-2 secondes
- Wikipedia: 0.5 secondes
- Google Custom Search: 0.5 secondes
- **Total actuellement:** ~3-5 secondes par claim
- **Solution:** Timeout + cache

### 3. Qualité variable des sources
- Wikipedia: peut avoir bias ou être incomplet
- Wolfram: ne traite que faits quantitatifs
- Google: snippets trop courts pour contexte
- **Solution:** Confidence scoring + fusion

---

## 🎯 Priorités pour amélioration

1. **HIGH** (Impact élevé, effort faible)
   - Augmenter à 3000 chars au lieu de 1200
   - Ajouter validation: page.content > 500 chars

2. **HIGH** (Impact élevé, effort moyen)
   - Améliorer stratégie Wolfram Alpha (plus agressif)
   - Ajouter timeout + fallback sur Google

3. **MEDIUM** (Impact moyen, effort élevé)
   - Multi-source fusion avec confidence scoring
   - Extraction de sections pertinentes

4. **LOW** (Nice-to-have)
   - Cache Redis pour claims fréquents
   - Fine-tuning DeBERTa sur domaine spécifique

---

## � Bug critique découvert: Pages "misinformation" inversent le verdict

### Exemple concret
```
Claim: "5G towers cause COVID-19"
Evidence Title: "5G misinformation"
DeBERTa Input: "5G misinformation [SEP] 5G towers cause COVID-19"
Result: ✓ SUPPORTED (99.6% confiance) ← INCORRECT!
```

### Le problème
Wikipedia retourne les pages "X misinformation" quand on cherche "X causes Y".

Exemple:
- Cherche: "5G causes COVID"
- Wikipedia retourne: "5G misinformation" (page qui DÉBUNKE, pas qui SOUTIENT)
- DeBERTa reçoit: Juste le titre vide "5G misinformation"
- Le modèle voit: "5G" vs "5G" = match = ENTAILMENT = SUPPORTED ✓ FAUX!

### Contenu réel manquant
Au lieu de recevoir:
```
"5G and COVID-19: There is no scientific link. 
COVID-19 is caused by a virus (SARS-CoV-2), which spreads through 
water droplets. 5G operates at radio frequencies and cannot transmit 
diseases. Many studies have debunked this conspiracy theory..."
```

On reçoit juste:
```
"5G misinformation"
```

Résultat: DeBERTa ne peut pas comprendre que c'est une **réfutation**, pas un soutien.

### Cas affectés
- "Moon landing was faked" → trouve "Moon landing conspiracy theories" → NOT_ENOUGH_INFO (faux)
- "COVID vaccines contain microchips" → trouve "Microchip conspiracy" page → peut inversion verdict
- "Flat earth" → trouve "Flat earth theory" misinformation page → verdict confus

### Solution rapide (à implémenter)
Détecter les pages "misinformation/conspiracy/hoax/debunked" et:
1. Soit les filtrer et relancer la recherche
2. Soit inverser la logique du verdict
3. Soit récupérer le contenu complet (et pas juste le titre) pour donner au modèle NLI du vrai contenu de débunking

---

## 📝 Prochaines étapes recommandées

Créer `evidence_retrieval_v2.py` avec:
```python
class EvidenceRetrieverV2(EvidenceRetriever):
    """Version améliorée avec meilleure extraction"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_min_length = 500  # Valider substantif
        self.content_max_length = 3000  # Augmenter de 1200
        self.cache = {}  # Cache simple
        self.debunking_keywords = [
            'misinformation', 'conspiracy', 'hoax', 'debunked', 
            'false claim', 'unfounded', 'myth', 'pseudoscience'
        ]
    
    def get_evidence(self, claim_text, language='en'):
        # 1. Wolfram (confiance 95%)
        # 2. Wikipedia enrichi (1200→3000 chars)
        # 3. Détecter pages "misinformation" et adapter la logique
        # 4. Entités + recherche
        # 5. Google (confiance 60%)
        # 6. Confidence scoring
```

