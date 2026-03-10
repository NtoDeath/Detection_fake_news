import wikipediaapi
import spacy
import requests

class EvidenceRetriever:

  def __init__(self, config_languages = {'en': 'en_core_web_sm', 'fr':'fr_core_news_sm', 'es':'es_core_news_sm'}):
    self.wiki_instances = {}
    self.nlp_models = {}

    user_agent = "FakeNewsDetectionProject/1.0 (contact@your-email.com)"

    for language, spacy_models in config_languages.items():
      self.wiki_instances[language] = wikipediaapi.Wikipedia(
                user_agent=user_agent,
                language=language

      )

      try:
        self.nlp_models[language] = spacy.load(spacy_models)
      except OSError:
        print(f"⚠️ Modèle {spacy_models} non trouvé. Lancez : !python -m spacy download {spacy_models}")

  def extract_keywords(self, text, language):
    nlp = self.nlp_models.get(language)
    if not nlp:
        return text  # Retour par défaut si le modèle manque

    doc = nlp(text)
        # Extraction des entités nommées (lieux, personnes, orgs)
    entities = [ent.text for ent in doc.ents]
    return " ".join(entities) if entities else text


  def get_evidence(self, claim_text, language='en'):
    wiki = self.wiki_instances.get(language)
    if not wiki: return None

    # ÉTAPE 1 : On extrait uniquement les entités (ex: "Eiffel Tower")
    entities = self.extract_keywords(claim_text, language)
    
    # ÉTAPE 2 : On essaie d'abord de chercher la page exacte de l'entité
    # C'est beaucoup plus sûr que de chercher la phrase avec les chiffres
    search_url = f"https://{language}.wikipedia.org/w/api.php"
    headers = {'User-Agent': 'FakeNewsDetectionProject/1.0 (contact@email.com)'}
    
    # On tente d'abord le titre exact (opensearch est parfait pour ça)
    params_opensearch = {
        "action": "opensearch",
        "search": entities,
        "limit": 1,
        "format": "json"
    }

    try:
        res = requests.get(search_url, params=params_opensearch, headers=headers)
        data = res.json()
        if data[1]:
            best_title = data[1][0]
        else:
            # Fallback : Recherche full-text si opensearch ne trouve rien
            params_search = {"action": "query", "list": "search", "srsearch": entities, "srlimit": 1, "format": "json"}
            res = requests.get(search_url, params=params_search, headers=headers)
            hits = res.json().get('query', {}).get('search', [])
            best_title = hits[0]['title'] if hits else None

        if best_title:
            print(f"🔍 [Wiki {language}] Page cible : {best_title}")
            page = wiki.page(best_title)
            if page.exists():
                return {"title": page.title, "content": page.summary[:1000], "url": page.fullurl}
                
    except Exception as e:
        print(f"Erreur : {e}")
    return None


  def search_multi_lingual(self, claim_text):
      results = {}
      for language in self.wiki_instances:
          evidence = self.get_evidence(claim_text, language)
          if evidence:
              results[language] = evidence
      return results

