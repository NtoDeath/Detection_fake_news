"""Claim detection module for knowledge-based fake news analysis.

Identifies check-worthy claims in text using two-stage pipeline:
1. Sentence segmentation (spaCy)
2. Checkworthiness classification (ALBERT)

A check-worthy claim is a statement that can be fact-checked.
Examples:
- "The Federal Reserve raised rates by 0.5%" (checkworthy)
- "This article is terrible" (not checkworthy, opinion)

Model
-----
ALBERT-base-v2 fine-tuned on checkworthiness detection.
Discriminates factual claims from opinions, greetings, meta-text.

Dependencies
------------
- transformers: HuggingFace pipeline for text classification
- spacy: sentence segmentation (en_core_web_sm model)
- pandas: data loading from JSONL format
"""

from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas


class ClaimDetector:
    """Detect check-worthy claims in text via sentence classification.
    
    Combines spaCy sentence segmentation with ALBERT checkworthiness model.
    
    Attributes
    ----------
    nlp : spacy.Language
        Loaded spaCy English model (en_core_web_sm).
    classifier : transformers.Pipeline
        ALBERT text classification model (checkworthiness).
    
    Notes
    -----
    Checkworthiness model trained on ClaimBuster dataset.
    LABEL_1 indicates check-worthy, LABEL_0 indicates not check-worthy.
    """
    
    def __init__(self) -> None:
        """Initialize sentence segmenter and checkworthiness classifier.
        
        Loads spaCy English model and downloads ALBERT from HuggingFace
        (first-time downloads are cached in ~/.cache/huggingface).
        
        Raises
        ------
        OSError
            If spaCy model download fails.
        """
        # Load spaCy English pipeline for sentence segmentation
        self.nlp: spacy.Language = spacy.load('en_core_web_sm')
        
        # Load ALBERT checkworthiness classifier from HuggingFace
        model_name: str = "tals/albert-base-v2-checkworthiness"
        self.classifier: 'transformers.Pipeline' = pipeline("text-classification", model=model_name)

    def process_text(self, text: str) -> List[str]:
        """Extract check-worthy claims from raw text.
        
        Pipeline:
        1. Segment text into sentences (spaCy)
        2. Classify each sentence as check-worthy or not (ALBERT)
        3. Return only check-worthy sentences
        
        Parameters
        ----------
        text : str
            Raw news article or paragraph.
        
        Returns
        -------
        list of str
            Check-worthy sentences (duplicates removed, whitespace trimmed).
        
        Notes
        -----
        Check-worthy = can be fact-checked (factual claim, not opinion/greeting).
        Empty result list if no claims detected.
        """
        # Stage 1: Sentence segmentation with spaCy dependency parser
        doc: 'spacy.Doc' = self.nlp(text)
        sentences: List[str] = [sent.text for sent in doc.sents]

        # Stage 2: Classify each sentence for checkworthiness
        results: List[Dict] = self.classifier(sentences)

        # Stage 3: Filter and collect check-worthy claims
        claims: List[str] = []
        for sentence, result in zip(sentences, results):
            # LABEL_1 = check-worthy, LABEL_0 = not check-worthy
            if result['label'] == 'LABEL_1':
                claims.append(sentence.strip())
        return claims

    def load_data(self, file_path: str) -> List[str]:
        """Load text corpus from JSONL file.
        
        Parameters
        ----------
        file_path : str
            Path to JSONL file (one JSON object per line).
        
        Returns
        -------
        list of str
            Raw text values from 'text' column of each JSON record.
        
        Notes
        -----
        Assumes 'text' column in JSON. Raises KeyError if missing.
        Example: {"id": 1, "text": "Article here..."}
        """
        df: pandas.DataFrame = pandas.read_json(file_path, lines=True)
        # Extract text column and convert to list
        return df['text'].tolist()