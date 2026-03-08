import re
import pandas as pd
import emoji
import spacy
from langdetect import detect
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
    def __init__(self):
        # Load heavy models only once upon object creation
        print("Loading linguistic models...")
        self.nlp_fr = spacy.load("fr_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm")
        self.spell_fr = SpellChecker(language='fr')
        self.spell_en = SpellChecker(language='en')
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        # Required for Scikit-Learn pipelines
        return self

    def _normalize_text(self, text):
        if not isinstance(text, str):
            return ""
            
        # Replace URLs with [URL] tag
        clean_text = re.sub(r'http\S+|www.\S+', '[URL]', text)
        
        # Replace Twitter/X mentions with [MENTION]
        clean_text = re.sub(r'@\w+', '[MENTION]', clean_text)
        
        # Replace numbers with [NB]
        clean_text = re.sub(r'\b\d+(?:[., :h]\d+)*\b', '[NB]', clean_text)    
        clean_text = re.sub(r'\b\d+\s*(?:min|lbs|kg|km|ml|ft|in|oz|°C|°F|°K|h|s|m|g|l|°|%|€|\$|£|¥)?(?!\w)', '[NB]', clean_text)

        return clean_text

    def _extract_metrics(self, raw_text):
        if not raw_text or len(raw_text.strip()) == 0:
            return {}

        metrics = {}

        metrics['has_hashtags'] = len(re.findall(r'#\w+', raw_text)) > 0
        metrics['has_mentions'] = len(re.findall(r'\[MENTION\]', raw_text)) > 0
        metrics['has_urls'] = len(re.findall(r'\[URL\]', raw_text)) > 0
        metrics['has_numbers'] = len(re.findall(r'\[NB\]', raw_text)) > 0
        
        hashtag_list = re.findall(r'#\w+', raw_text)
        
        # Clean tags to avoid false spelling mistakes
        raw_text = re.sub(r'\s*\[(?:MENTION|URL|NB)\]\s*', ' ', raw_text)
        raw_text = raw_text.replace('#', '').strip()
        
        metrics['total_characters'] = len(raw_text)
        num_letters = sum(1 for c in raw_text if c.isalpha())
        num_uppercase = sum(1 for c in raw_text if c.isupper())
        
        metrics['uppercase_ratio'] = num_uppercase / num_letters if num_letters > 0 else 0
        
        num_exclamations = raw_text.count('!')
        num_questions = raw_text.count('?')
        metrics['aggressive_punct_density'] = (num_exclamations + num_questions) / len(raw_text) if len(raw_text) > 0 else 0

        try:
            language = detect(raw_text)
        except:
            language = 'en'
            
        if language == 'fr':
            doc = self.nlp_fr(raw_text)
            emoji_language = 'fr'
        else:
            doc = self.nlp_en(raw_text)
            emoji_language = 'en'
        
        # Filter to keep only real words (exclude punctuation and spaces)
        real_words = [token for token in doc if not token.is_punct and not token.is_space]
        num_words = len(real_words)
        num_sentences = len(list(doc.sents))
        
        if num_words == 0:
            return metrics

        metrics['num_words'] = num_words
        metrics['num_sentences'] = num_sentences

        unique_words = set([token.text.lower() for token in real_words])
        metrics['lexical_richness'] = len(unique_words) / num_words
        
        total_word_length = sum(len(token.text) for token in real_words)
        metrics['avg_word_length'] = total_word_length / num_words
        metrics['avg_sentence_length'] = num_words / num_sentences if num_sentences > 0 else 0

        # POS tagging ratios (ADJ = Adjective, PRON = Pronoun, ADV = Adverb, VERB = Verb)
        metrics['adjective_ratio'] = sum(1 for token in real_words if token.pos_ == "ADJ") / num_words
        metrics['pronoun_ratio'] = sum(1 for token in real_words if token.pos_ == "PRON") / num_words
        metrics['adverb_ratio'] = sum(1 for token in real_words if token.pos_ == "ADV") / num_words
        metrics['verb_ratio'] = sum(1 for token in real_words if token.pos_ == "VERB") / num_words
        
        metrics['num_hashtags'] = len(hashtag_list)
        metrics['num_emojis'] = emoji.emoji_count(raw_text)

        # Spell checking (ignoring proper nouns)
        words_to_check = [token.text.lower() for token in real_words if token.pos_ != "PROPN"]
        
        if language == 'fr':
            spelling_mistakes = self.spell_fr.unknown(words_to_check)
        else:
            spelling_mistakes = self.spell_en.unknown(words_to_check)
            
        metrics['spelling_mistake_ratio'] = len(spelling_mistakes) / num_words if num_words > 0 else 0

        text_emojis_words = emoji.demojize(raw_text, language=emoji_language)
        text_emojis_words = re.sub(r':([^\s:]+):', r'\1', text_emojis_words)
        text_emojis_words = text_emojis_words.replace('_', ' ')
        
        # Sentiment
        if language == 'fr':
            blob = TextBlob(text_emojis_words, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            metrics['sentiment_polarity'] = blob.sentiment[0]
            metrics['subjectivity'] = blob.sentiment[1]
        else:
            blob = TextBlob(text_emojis_words)
            metrics['sentiment_polarity'] = self.vader_analyzer.polarity_scores(text_emojis_words)['compound']
            metrics['subjectivity'] = blob.sentiment.subjectivity

        return metrics


def transform(self, X, y=None):
        features = []
        for text in tqdm(X, desc="Extracting Style Metrics"):
            clean_text = self._normalize_text(text)
            vector = self._extract_metrics(clean_text)
            features.append(vector)

        df_features = pd.DataFrame(features)
        
        return df_features.fillna(0)