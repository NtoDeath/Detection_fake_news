"""Style-based fake news detection module.

Extracts linguistic and stylometric features from text to detect fake news.
Features include punctuation density, sentiment, word metrics, spelling errors,
and structural characteristics. Supports bilingual processing (English, French).

Notes
-----
Requires: spacy models (en_core_web_sm, fr_core_news_sm), NLTK VADER lexicon.
Implements Scikit-Learn transformer interface (fit, transform).
Depends on: pandas, spacy, textblob, nltk, langdetect, emoji, spellchecker.
"""

import re
from typing import Dict, Optional, List, Any
import pandas as pd
import emoji
import spacy
from spacy.language import Language
from langdetect import detect, LangDetectException
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from spellchecker import SpellChecker
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
# import spacy.cli # uncomment for the first time
# spacy.cli.download("fr_core_news_sm") # uncomment for the first time
# nltk.download('vader_lexicon', quiet=True) # uncomment for the first time

class StyleExtractor(BaseEstimator, TransformerMixin):
    """StyleExtractor: Bilingual linguistic feature extraction.
    
    Computes 20+ stylometric features from text. Tokenizes with spaCy,
    detects language, extracts POS tags, sentiment, and text structure metrics.
    
    Attributes
    ----------
    nlp_fr : Language
        spaCy French pipeline (fr_core_news_sm).
    nlp_en : Language
        spaCy English pipeline (en_core_web_sm).
    spell_fr : SpellChecker
        French spell checker.
    spell_en : SpellChecker
        English spell checker.
    vader_analyzer : SentimentIntensityAnalyzer
        NLTK VADER for English sentiment analysis.
    """
    def __init__(self) -> None:
        """Initialize StyleExtractor with linguistic models.
        
        Loads spaCy pipelines (English, French), initializes spell checkers,
        and VADER sentiment analyzer. Models are cached after first load.
        
        Notes
        -----
        First initialization loads 300+ MB of language models.
        Silence spaCy warnings with: python -m spacy validate
        """
        print("Loading linguistic models...")
        self.nlp_fr: Language = spacy.load("fr_core_news_sm")
        self.nlp_en: Language = spacy.load("en_core_web_sm")
        self.spell_fr: SpellChecker = SpellChecker(language='fr')
        self.spell_en: SpellChecker = SpellChecker(language='en')
        self.vader_analyzer: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

    def fit(self, X: Any, y: Optional[Any] = None) -> 'StyleExtractor':
        """Fit method (no-op for compatibility with Scikit-Learn).
        
        Parameters
        ----------
        X : array-like
            Input data (unused).
        y : array-like, optional
            Target data (unused).
        
        Returns
        -------
        StyleExtractor
            Self instance.
        
        Notes
        -----
        Required for Scikit-Learn pipeline compatibility.
        No actual fitting occurs; models are pre-loaded in __init__.
        """
        return self

    def _normalize_text(self, text: Optional[str]) -> str:
        """Normalize text by masking entities and standardizing format.
        
        Replaces URLs, mentions, numbers with tokens. Preserves sentence structure
        for linguistic analysis. Critical preprocessing step before metrics extraction.
        
        Parameters
        ----------
        text : str or None
            Input text to normalize.
        
        Returns
        -------
        str
            Normalized text with masked entities.
        
        Notes
        -----
        Entity masking: URLs=[URL], mentions=[MENTION], numbers=[NB].
        Handles currency symbols, units, timestamps. Empty input returns empty string.
        """
        if not isinstance(text, str):
            return ""
            
        # Replace URLs (http, ftp, www) with [URL] token
        clean_text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        
        # Replace Twitter/X mentions with [MENTION] token
        clean_text = re.sub(r'@\w+', '[MENTION]', clean_text)
        
        # Replace numbers with [NB] token (includes decimals, currency, units)
        clean_text = re.sub(r'\b\d+(?:[., :h]\d+)*\b', '[NB]', clean_text)    
        clean_text = re.sub(r'\b\d+\s*(?:min|lbs|kg|km|ml|ft|in|oz|°C|°F|°K|h|s|m|g|l|°|%|€|\$|£|¥)?(?!\w)', '[NB]', clean_text)

        return clean_text

    def _extract_metrics(self, raw_text: str) -> Dict[str, Any]:
        """Extract 20+ stylometric features from normalized text.
        
        Computes linguistic features: punctuation density, sentiment, word metrics,
        POS ratios, spelling errors, structural properties. Requires normalized text.
        
        Parameters
        ----------
        raw_text : str
            Normalized text (after _normalize_text).
        
        Returns
        -------
        dict
            Feature dictionary with keys:
            - Boolean flags: has_hashtags, has_mentions, has_urls, has_numbers
            - Character metrics: total_characters, uppercase_ratio, aggressive_punctuation_density
            - Lexical metrics: word_count, sentence_count, lexical_richness, avg_word_length
            - Syntactic metrics: adjective_ratio, pronoun_ratio, adverb_ratio, verb_ratio
            - Social metrics: hashtag_count, emoji_count
            - Text quality: spelling_mistake_ratio
            - Sentiment: sentiment_polarity, subjectivity
        
        Raises
        ------
        No explicit exceptions; empty text returns empty dict.
        
        Notes
        -----
        Language detection (French/English) determines POS tagger and sentiment analyzer.
        Fallback to English if detection fails. Sentiment uses TextBlob (FR) or VADER (EN).
        """
        if not raw_text or len(raw_text.strip()) == 0:
            return {}

        metrics: Dict[str, Any] = {}

        # Boolean flags: presence of social media markers and entities
        metrics['has_hashtags'] = len(re.findall(r'#\w+', raw_text)) > 0
        metrics['has_mentions'] = len(re.findall(r'\[MENTION\]', raw_text)) > 0
        metrics['has_urls'] = len(re.findall(r'\[URL\]', raw_text)) > 0
        metrics['has_numbers'] = len(re.findall(r'\[NB\]', raw_text)) > 0
        
        hashtag_list = re.findall(r'#\w+', raw_text)
        
        # Remove entity tokens and hashtags for cleaner linguistic analysis
        raw_text_clean = re.sub(r'\s*\[(?:MENTION|URL|NB)\]\s*', ' ', raw_text)
        raw_text_clean = raw_text_clean.replace('#', '').strip()
        
        # Character and case metrics
        metrics['total_characters'] = len(raw_text_clean)
        num_letters = sum(1 for c in raw_text_clean if c.isalpha())
        num_uppercase = sum(1 for c in raw_text_clean if c.isupper())
        
        metrics['uppercase_ratio'] = num_uppercase / num_letters if num_letters > 0 else 0
        
        # Word-level capitalization (often indicates fake news tone)
        words = raw_text_clean.split()
        num_fully_uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        metrics['fully_uppercase_words_ratio'] = num_fully_uppercase_words / len(words) if len(words) > 0 else 0

        # Aggressive punctuation metric (! and ? indicate emotional language)
        num_exclamations = raw_text_clean.count('!')
        num_questions = raw_text_clean.count('?')
        metrics['aggressive_punctuation_density'] = (num_exclamations + num_questions) / len(raw_text_clean) if len(raw_text_clean) > 0 else 0

        # Language detection for appropriate NLP pipeline
        try:
            language = detect(raw_text_clean)
        except LangDetectException:
            language = 'en'  # Default to English if detection fails
            
        # Apply language-specific tokenizer and tagger
        if language == 'fr':
            doc = self.nlp_fr(raw_text_clean)
            emoji_language = 'fr'
        else:
            doc = self.nlp_en(raw_text_clean)
            emoji_language = 'en'
        
        # Extract words (exclude punctuation and whitespace)
        real_words = [token for token in doc if not token.is_punct and not token.is_space]
        num_words = len(real_words)
        num_sentences = len(list(doc.sents))
        
        if num_words == 0:
            return metrics

        metrics['word_count'] = num_words
        metrics['sentence_count'] = num_sentences

        # Lexical diversity (type-token ratio)
        unique_words = set([token.text.lower() for token in real_words])
        metrics['lexical_richness'] = len(unique_words) / num_words
        
        # Word length and sentence complexity metrics
        total_word_length = sum(len(token.text) for token in real_words)
        metrics['avg_word_length'] = total_word_length / num_words
        metrics['avg_sentence_length'] = num_words / num_sentences if num_sentences > 0 else 0

        # Part-of-speech ratios (linguistic style markers)
        metrics['adjective_ratio'] = sum(1 for token in real_words if token.pos_ == "ADJ") / num_words
        metrics['pronoun_ratio'] = sum(1 for token in real_words if token.pos_ == "PRON") / num_words
        metrics['adverb_ratio'] = sum(1 for token in real_words if token.pos_ == "ADV") / num_words
        metrics['verb_ratio'] = sum(1 for token in real_words if token.pos_ == "VERB") / num_words
        
        # Social media and expression metrics
        metrics['hashtag_count'] = len(hashtag_list)
        metrics['emoji_count'] = emoji.emoji_count(raw_text)

        # Spelling quality metric (ignorant of proper nouns)
        words_to_check = [token.text.lower() for token in real_words if token.pos_ != "PROPN"]
        
        if language == 'fr':
            spelling_mistakes = self.spell_fr.unknown(words_to_check)
        else:
            spelling_mistakes = self.spell_en.unknown(words_to_check)
            
        metrics['spelling_mistake_ratio'] = len(spelling_mistakes) / num_words if num_words > 0 else 0

        # Convert emoji to text representation for sentiment analysis
        text_emojis_words = emoji.demojize(raw_text_clean, language=emoji_language)
        text_emojis_words = re.sub(r':([^\s:]+):', r'\1', text_emojis_words)
        text_emojis_words = text_emojis_words.replace('_', ' ')
        
        # Sentiment and subjectivity (language-specific analyzers)
        if language == 'fr':
            blob = TextBlob(text_emojis_words, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            metrics['sentiment_polarity'] = blob.sentiment[0]
            metrics['subjectivity'] = blob.sentiment[1]
        else:
            blob = TextBlob(text_emojis_words)
            metrics['sentiment_polarity'] = self.vader_analyzer.polarity_scores(text_emojis_words)['compound']
            metrics['subjectivity'] = blob.sentiment.subjectivity

        return metrics

    def transform(self, X: List[str], y: Optional[Any] = None) -> pd.DataFrame:
        """Transform corpus to stylometric feature matrix.
        
        Applies _normalize_text and _extract_metrics to each text in corpus.
        Returns DataFrame with computed features per row.
        
        Parameters
        ----------
        X : list of str
            Input texts to process.
        y : array-like, optional
            Target data (unused).
        
        Returns
        -------
        DataFrame
            Shape (n_samples, n_features). Columns are feature names.
            Missing values filled with 0. Columns with all NaNs removed.
        
        Notes
        -----
        Processing time scales ~1-3 seconds per 1000 texts (GPU-dependent).
        Progress bar shown via tqdm.
        """
        features = []
        for text in tqdm(X, desc="Extracting Style Metrics"):
            clean_text = self._normalize_text(text)
            vector = self._extract_metrics(clean_text)
            features.append(vector)

        df_features = pd.DataFrame(features)
        
        # Remove columns that are entirely NaN
        df_features = df_features.dropna(axis=1, how='all')
        
        return df_features.fillna(0)