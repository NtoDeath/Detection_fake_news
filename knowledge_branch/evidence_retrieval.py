import wikipediaapi
import spacy
import requests

class EvidenceRetriever:
  def __init__(self, google_api_key=None, google_cse_id=None, wolfram_app_id=None,
                 config_languages={'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm'}):
      self.wiki_instances = {}
      self.nlp_models = {}
      self.google_api_key = google_api_key
      self.google_cse_id = google_cse_id
      self.wolfram_app_id = wolfram_app_id

      user_agent = "FakeNewsDetectionProject/1.0 "

      for language, spacy_models in config_languages.items():
        self.wiki_instances[language] = wikipediaapi.Wikipedia(
            user_agent=user_agent, language=language)

        try:
          self.nlp_models[language] = spacy.load(spacy_models)

        except OSError:
          print(f"⚠️ Modèle {spacy_models} manquant.")

  def extract_entities(self, text, language):
      nlp = self.nlp_models.get(language)
      if not nlp:
        return text
      doc = nlp(text)
      entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'LOC', 'GPE', 'ORG', 'FAC']]
      return " ".join(entities) if entities else text

  def get_wolfram_evidence(self, claim):
    if not self.wolfram_app_id:
      return None
    url = "http://api.wolframalpha.com/v1/result"
    parameters = {
        "appid": self.wolfram_app_id, "i": claim
    }
    try:
      response = requests.get(url, params=parameters)
      if response.status_code == 200:
          return {"title": "WolframAlpha", "content": response.text, "url": "https://www.wolframalpha.com"}
    except: return None

  def get_google_politifact_evidence(self, claim):
    if not self.google_api_key:
      return None
    query = f"site:politifact.com OR site:lemonde.fr {claim}"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": self.google_api_key, "cx": self.google_cse_id, "q": query}

    try:
      response = requests.get(url, params=params).json()
      if 'items' in response:
        return {"title": response['items'][0]['title'], "content": response['items'][0]['snippet'], "url": response['items'][0]['link']}
    except:
      return None

  def get_evidence(self, claim_text, language='en'):

    subject_query = self.extract_entities(claim_text, language)

    evidence = self.get_wolfram_evidence(subject_query)
    if evidence:
      return evidence

    evidence = self.get_google_politifact_evidence(subject_query)
    if evidence: return evidence

    wiki = self.wiki_instances.get(language)

    if wiki:
      search_url = f"https://{language}.wikipedia.org/w/api.php"
      # Ajoute un User-Agent clair ici
      headers = {"User-Agent": "FactCheckerProject/1.0 (votre_email@example.com)"}
      
      params = {
          "action": "opensearch", 
          "search": subject_query, 
          "limit": 1, 
          "format": "json"
      }
      
      response = requests.get(search_url, params=params, headers=headers)
      
      # Sécurité : on vérifie que la réponse est OK (code 200)
      if response.status_code == 200:
          try:
              data = response.json()
              if len(data) > 1 and data[1]:
                  page = wiki.page(data[1][0])
                  if page.exists():
                      return {"title": page.title, "content": page.summary[:1200], "url": page.fullurl}
          except Exception as e:
              print(f"Erreur lors du décodage JSON : {e}")
      else:
          print(f"Erreur Wikipedia : Code {response.status_code}")
          
    return None