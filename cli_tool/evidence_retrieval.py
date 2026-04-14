"""
Evidence retrieval from Wikipedia, Wolfram Alpha, and Google Search.
"""

import wikipedia
import requests
from typing import Optional, Dict, Any


class EvidenceRetriever:
    """Retrieve evidence for fact-checking from various sources"""
    
    def __init__(self, google_api_key=None, google_cse_id=None, wolfram_app_id=None,
                 config_languages={'en': 'en_core_web_sm'}):
        self.nlp_models = {}
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.wolfram_app_id = wolfram_app_id

        # Try to load spaCy models for NER
        try:
            import spacy
            for language, spacy_model in config_languages.items():
                try:
                    self.nlp_models[language] = spacy.load(spacy_model)
                except OSError:
                    self.nlp_models[language] = None
        except ImportError:
            pass

    def extract_entities(self, text: str, language: str) -> str:
        """Extract named entities from text using spaCy"""
        nlp = self.nlp_models.get(language)
        if not nlp:
            return text
        
        try:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'LOC', 'GPE', 'ORG', 'FAC']]
            return " ".join(entities) if entities else text
        except:
            return text

    def get_wolfram_evidence(self, claim: str) -> Optional[Dict[str, Any]]:
        """Get evidence from Wolfram Alpha"""
        if not self.wolfram_app_id:
            return None
        
        url = "http://api.wolframalpha.com/v1/result"
        parameters = {
            "appid": self.wolfram_app_id,
            "i": claim
        }
        
        try:
            response = requests.get(url, params=parameters, timeout=5)
            if response.status_code == 200 and response.text and len(response.text) > 10:
                return {
                    "title": "WolframAlpha",
                    "content": response.text,
                    "url": "https://www.wolframalpha.com",
                    "source": "WolframAlpha"
                }
        except:
            pass
        
        return None

    def get_google_evidence(self, claim: str) -> Optional[Dict[str, Any]]:
        """Get evidence from Google Custom Search"""
        if not self.google_api_key or not self.google_cse_id:
            return None
        
        query = f"site:politifact.com OR site:snopes.com {claim}"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query
        }

        try:
            response = requests.get(url, params=params, timeout=5).json()
            if 'items' in response and len(response['items']) > 0:
                item = response['items'][0]
                return {
                    "title": item.get('title', ''),
                    "content": item.get('snippet', ''),
                    "url": item.get('link', '')
                }
        except:
            pass
        
        return None

    def get_evidence(self, claim_text: str, language: str = 'en') -> Optional[Dict[str, Any]]:
        """
        Get evidence for a claim from multiple sources.
        
        Strategy:
        1. Try Wolfram Alpha (factual queries)
        2. Try Wikipedia (general knowledge)
        3. Try Google Search (fact-checking sites)
        """
        
        # 1. Try Wolfram Alpha
        evidence = self.get_wolfram_evidence(claim_text)
        if evidence and len(evidence.get('content', '')) > 150:
            evidence['source'] = 'WolframAlpha'
            return evidence

        # 2. Try Wikipedia
        def wiki_query(search_string: str) -> Optional[Dict[str, Any]]:
            try:
                # Search for the page
                search_results = wikipedia.search(search_string, results=1)
                if not search_results:
                    return None
                
                # Get the first result
                page_title = search_results[0]
                page = wikipedia.page(page_title, auto_suggest=False)
                
                content = page.content[:1200]  # Get first 1200 chars
                if len(content.strip()) < 50:
                    return None
                    
                return {
                    'title': page.title,
                    'content': content,
                    'url': page.url,
                    'source': 'Wikipedia'
                }
            except (wikipedia.exceptions.DisambiguationError, 
                    wikipedia.exceptions.PageError,
                    Exception):
                return None

        # Try with full claim first
        wiki_evidence = wiki_query(claim_text)
        if wiki_evidence:
            return wiki_evidence

        # Try with extracted entities
        keywords = self.extract_entities(claim_text, language)
        if keywords != claim_text:
            wiki_evidence = wiki_query(keywords)
            if wiki_evidence:
                return wiki_evidence

        # 3. Try Google Search
        google_evidence = self.get_google_evidence(claim_text)
        if google_evidence:
            google_evidence['source'] = 'GoogleSearch'
            return google_evidence

        return None
