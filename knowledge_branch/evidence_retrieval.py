"""Evidence retrieval for fact-checking (knowledge-based verification).

Retrieves supporting or contradicting evidence from multiple sources:
1. Wikipedia (bilingual: EN/FR)
2. WolframAlpha (mathematical/factual queries)
3. Google Custom Search (PolitiFact, Le Monde fact-checking articles)

Pipeline
--------
Input: Extracted claim
  │
  ├─→ Entity extraction (NER): extract key persons, organizations, locations
  │    (filters: PERSON, LOC, GPE, ORG, FAC)
  │
  ├─→ Query Wikipedia in claim language
  │    Search on key entities + claim terms
  │
  ├─→ Query WolframAlpha (for numbers, dates, statistics)
  │
  └─→ Query fact-checking sources (PolitiFact, Le Monde)
       Via Google Custom Search API

Output: Evidence dictionary {title, content, url, source}

Dependencies
------------
- wikipediaapi: Wikipedia content retrieval
- spacy: Named Entity Recognition (en_core_web_sm, fr_core_news_sm)
- requests: HTTP requests to external APIs
- Google API: Requires api_key and custom_search_engine_id
- WolframAlpha API: Requires app_id
"""

import wikipediaapi
import spacy
import requests
from typing import Dict, List, Optional, Any


class EvidenceRetriever:
    """Retrieve evidence from multiple sources to verify claims.
    
    Parameters
    ----------
    google_api_key : str, optional
        Google Custom Search API key.
    google_cse_id : str, optional
        Google Custom Search Engine ID.
    wolfram_app_id : str, optional
        WolframAlpha API app ID.
    config_languages : dict
        Language -> spaCy model name mapping.
        Default: {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm'}
    
    Attributes
    ----------
    wiki_instances : dict
        Wikipedia API instances per language.
    nlp_models : dict
        spaCy language models per language.
    google_api_key : str or None
        Google Custom Search credentials.
    google_cse_id : str or None
    wolfram_app_id : str or None
    
    Notes
    -----
    Bilingual support: English and French knowledge bases.
    API errors are caught and return None (graceful degradation).
    """
    
    def __init__(
        self, 
        google_api_key: Optional[str] = None, 
        google_cse_id: Optional[str] = None, 
        wolfram_app_id: Optional[str] = None,
        config_languages: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize evidence retriever with multi-source backends.
        
        Parameters
        ----------
        google_api_key : str, optional
            Enable Google Custom Search queries.
        google_cse_id : str, optional
            Custom search engine ID (configured on console.cloud.google.com).
        wolfram_app_id : str, optional
            Enable WolframAlpha queries.
        config_languages : dict, optional
            Custom language config. Default: EN and FR.
        
        Notes
        -----
        Missing spaCy models logged as warnings (doesn't crash initialization).
        First-time spaCy model downloads require python -m spacy download.
        """
        self.wiki_instances: Dict[str, Any] = {}
        self.nlp_models: Dict[str, Any] = {}
        self.google_api_key: Optional[str] = google_api_key
        self.google_cse_id: Optional[str] = google_cse_id
        self.wolfram_app_id: Optional[str] = wolfram_app_id

        # Default language configuration: English + French
        if config_languages is None:
            config_languages = {
                'en': 'en_core_web_sm',
                'fr': 'fr_core_news_sm'
            }

        # Custom user agent for API requests (best practice)
        user_agent: str = "FakeNewsDetectionProject/1.0 "

        # Initialize Wikipedia API for each configured language
        for language, spacy_model_name in config_languages.items():
            # Create Wikipedia instance with language parameter
            self.wiki_instances[language] = wikipediaapi.Wikipedia(
                user_agent=user_agent, language=language)

            # Load spaCy language model for NER
            try:
                self.nlp_models[language] = spacy.load(spacy_model_name)
            except OSError:
                print(f"⚠️ Modèle {spacy_model_name} manquant.")

    def extract_entities(self, text: str, language: str) -> str:
        """Extract named entities from text for targeted searches.
        
        Uses spaCy NER to identify persons, organizations, locations.
        Only PERSON, LOC, GPE, ORG, FAC are kept (ignore others).
        
        Parameters
        ----------
        text : str
            Raw text to extract entities from.
        language : str
            Language code ('en' or 'fr').
        
        Returns
        -------
        str
            Space-separated entity names (e.g., "Joe Biden Washington DC").
            Falls back to original text if model unavailable.
        
        Notes
        -----
        Entity types filtered:
        - PERSON: people names
        - LOC: non-political locations
        - GPE: countries, cities, states (political entities)
        - ORG: companies, institutions, organizations
        - FAC: buildings, facilities
        """
        # Get spaCy model for language (None if unavailable)
        nlp: Optional[Any] = self.nlp_models.get(language)
        if not nlp:
            # Fallback: return original text if NER unavailable
            return text
        
        # Run spaCy processing pipeline (tokenization, POS, NER)
        doc: 'spacy.Doc' = nlp(text)
        
        # Extract only fact-checkable entity types
        entities: List[str] = [
            ent.text for ent in doc.ents 
            if ent.label_ in ['PERSON', 'LOC', 'GPE', 'ORG', 'FAC']
        ]
        
        # Return concatenated entities or original text if none found
        return " ".join(entities) if entities else text

    def get_wolfram_evidence(self, claim: str) -> Optional[Dict[str, str]]:
        """Query WolframAlpha for mathematical/factual answers.
        
        Asks WolframAlpha computation engine simple factual questions.
        Works well for: statistics, dates, mathematical facts, scientific info.
        
        Parameters
        ----------
        claim : str
            Factual query (e.g., "population of France").
        
        Returns
        -------
        dict or None
            Evidence dict with keys: title, content, url.
            Returns None if api_id not configured or request fails.
        
        Notes
        -----
        Requires wolfram_app_id set at initialization.
        API endpoint: api.wolframalpha.com/v1/result
        Graceful error handling: returns None on HTTP errors.
        """
        # Skip if WolframAlpha credentials not configured
        if not self.wolfram_app_id:
            return None
        
        # WolframAlpha API endpoint for simple text responses
        url: str = "http://api.wolframalpha.com/v1/result"
        parameters: Dict[str, str] = {
            "appid": self.wolfram_app_id,
            "i": claim  # Input query
        }
        
        try:
            # Make HTTP request
            response: 'requests.Response' = requests.get(url, params=parameters)
            
            # Successful response (HTTP 200)
            if response.status_code == 200:
                return {
                    "title": "WolframAlpha",
                    "content": response.text,
                    "url": "https://www.wolframalpha.com"
                }
        except Exception:
            # Network errors, timeouts, etc. - fail gracefully
            return None

    def get_google_politifact_evidence(self, claim: str) -> Optional[Dict[str, str]]:
        """Search fact-checking sites (PolitiFact, Le Monde) via Google Custom Search.
        
        Queries Google Custom Search API for claims already fact-checked
        by PolitiFact (English) and Le Monde (French).
        
        Parameters
        ----------
        claim : str
            Claim text to search in fact-checking sources.
        
        Returns
        -------
        dict or None
            First search result with keys: title, content (snippet), url.
            Returns None if API key not configured or no results found.
        
        Notes
        -----
        Limits search to: "site:politifact.com OR site:lemonde.fr"
        Requires google_api_key and google_cse_id set at initialization.
        Only returns first result (highest relevance).
        """
        # Skip if Google credentials not configured
        if not self.google_api_key:
            return None
        
        # Construct search query targeting fact-checking sites
        query: str = f"site:politifact.com OR site:lemonde.fr {claim}"
        url: str = "https://www.googleapis.com/customsearch/v1"
        params: Dict[str, str] = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query
        }

        try:
            # Make Custom Search API request
            response_json: Dict = requests.get(url, params=params).json()
            
            # Parse results
            if 'items' in response_json:
                first_result: Dict = response_json['items'][0]
                return {
                    "title": first_result['title'],
                    "content": first_result['snippet'],  # HTML snippet
                    "url": first_result['link']
                }
        except Exception:
            # JSON parse errors, missing keys, network errors - fail gracefully
            return None

            # JSON parse errors, missing keys, network errors - fail gracefully
            return None

    def get_evidence(self, claim_text: str, language: str = 'en') -> Optional[Dict[str, str]]:
        """Retrieve evidence from multiple sources for a claim.
        
        Orchestrates multi-source evidence retrieval:
        1. Try WolframAlpha (for facts, numbers, dates)
        2. Try Google Custom Search (fact-checking sites)
        3. Try Wikipedia search (general knowledge)
        
        Parameters
        ----------
        claim_text : str
            Claim to search evidence for.
        language : str
            Language code: 'en' (English) or 'fr' (French).
        
        Returns
        -------
        dict or None
            Evidence {title, content, url} from first available source.
            Returns None if no sources configured or no results found.
        
        Notes
        -----
        Tries sources in order of specificity: WolframAlpha → PolitiFact → Wikipedia
        Wikipedia fallback searches parsed entities extracted from claim.
        """
        # Extract key entities for targeted searches (persons, orgs, locations)
        subject_query: str = self.extract_entities(claim_text, language)

        # Try WolframAlpha (best for factual, mathematical claims)
        evidence: Optional[Dict] = self.get_wolfram_evidence(subject_query)
        if evidence:
            return evidence

        # Try fact-checking source (PolitiFact, Le Monde)
        evidence: Optional[Dict] = self.get_google_politifact_evidence(subject_query)
        if evidence:
            return evidence

        # Fallback: Wikipedia search on extracted entities
        wiki: Optional[Any] = self.wiki_instances.get(language)

        if wiki:
            # Wikipedia search via OpenSearch API (faster than direct page lookup)
            search_url: str = f"https://{language}.wikipedia.org/w/api.php"
            # Custom User-Agent (best practice for API requests)
            headers: Dict[str, str] = {"User-Agent": "FactCheckerProject/1.0 (votre_email@example.com)"}
            
            # OpenSearch parameters
            params: Dict[str, str] = {
                "action": "opensearch", 
                "search": subject_query, 
                "limit": 1,  # Get only top result
                "format": "json"
            }
            
            # Make request to Wikipedia API
            response: 'requests.Response' = requests.get(search_url, params=params, headers=headers)
            
            # Check for successful response
            if response.status_code == 200:
                try:
                    data: List = response.json()
                    # OpenSearch response format: [query, [titles], [descriptions], [urls]]
                    if len(data) > 1 and data[1]:
                        # Get Wikipedia page from first result
                        page: 'wikipediaapi.WikipediaPage' = wiki.page(data[1][0])
                        if page.exists():
                            return {
                                "title": page.title,
                                "content": page.summary[:1200],  # Truncate to 1200 chars
                                "url": page.fullurl
                            }
                except Exception as e:
                    print(f"Erreur lors du décodage JSON : {e}")
            else:
                print(f"Erreur Wikipedia : Code {response.status_code}")
          
        return None