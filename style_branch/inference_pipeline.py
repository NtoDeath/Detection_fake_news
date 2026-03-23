"""Production inference pipeline for fake news detection.

Implements Y-architecture ensemble combining:
- StyleExtractor: 20+ bilingual stylometric features
- RoBERTa: semantic understanding (fine-tuned)
- Random Forest: classification on merged feature space

Class
-----
FakeNewsDetector: Complete pipeline from text to prediction.

Usage
-----
```python
detector = FakeNewsDetector()
result = detector.analyze("news text here")
print(result['is_fake_news'], result['rf_confidence'])
```

Architecture Diagram::

    Raw Text
      │
      ├───→ StyleExtractor (20+ metrics)
      │          │
      │          ├─→ punctuation, word/sentence stats
      │          ├─→ sentiment, subjectivity
      │          ├─→ readability, spelling
      │          └─→ social media indicators
      │
      ├───→ RoBERTa (semantic)
      │          │
      │          └─→ fine-tuned fake news classification
      │               + probability score
      │
      └──→ Feature Merging
                  │
                  └──→ Random Forest Classifier
                       (trained on merged features)
                       │
                       └──→ Final Prediction:
                           - is_fake_news: bool
                           - rf_confidence: float (0-100)
                           - roberta_opinion: float (0-100)

Dependencies
------------
- torch: RoBERTa inference
- transformers: tokenizer + model
- joblib: model serialization
- style_extractor: StyleExtractor class
- scipy: softmax normalization
"""

import torch
import joblib
import pandas as pd
import os
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from style_extractor import StyleExtractor


class FakeNewsDetector:
    """Y-architecture ensemble for fake news detection.
    
    Combines three models:
    1. StyleExtractor: computes 20+ stylometric features
    2. RoBERTa: semantic model (fine-tuned on fake news)
    3. Random Forest: ensemble classifier on merged features
    
    Parameters
    ----------
    roberta_path : str
        Directory containing fine-tuned RoBERTa checkpoint.
    rf_path : str
        Path to trained Random Forest model (pickle format).
    
    Attributes
    ----------
    tokenizer : AutoTokenizer
        RoBERTa tokenizer.
    roberta_model : AutoModelForSequenceClassification
        Fine-tuned RoBERTa (2 labels: true/fake).
    rf_judge : RandomForestClassifier
        Ensemble classifier (scikit-learn).
    style_extractor : StyleExtractor
        Bilingual feature extractor.
    expected_columns : Index
        Feature names required by Random Forest.
    
    Raises
    ------
    FileNotFoundError
        If roberta_path or rf_path missing.
    
    Notes
    -----
    Production-ready: loads models into RAM on __init__.
    Memory footprint: ~500MB-1GB (BERT + RF + extractors).
    """
    
    def __init__(self, roberta_path: str = "./roberta_fine_tunned", rf_path: str = "./results/best_model.pkl") -> None:
        """Initialize detector by loading three component models.
        
        Parameters
        ----------
        roberta_path : str
            Directory path to fine-tuned RoBERTa checkpoint.
        rf_path : str
            File path to serialized Random Forest model.
        
        Raises
        ------
        FileNotFoundError
            If roberta_path or rf_path not found.
        """
        print("Initializing Detector (Loading into RAM...)")
        
        # Validate model paths exist before loading
        if not os.path.exists(roberta_path):
            raise FileNotFoundError(
                f"Missing model files! Ensure '{roberta_path}' exists."
            )
        if not os.path.exists(rf_path):
            raise FileNotFoundError(
                f"Missing model files! Ensure '{rf_path}' exists."
            )

        # Load RoBERTa tokenizer and model
        self.tokenizer: 'AutoTokenizer' = AutoTokenizer.from_pretrained(roberta_path)
        self.roberta_model: 'AutoModelForSequenceClassification' = AutoModelForSequenceClassification.from_pretrained(roberta_path)
        
        # Load trained Random Forest classifier
        self.rf_judge: 'RandomForestClassifier' = joblib.load(rf_path)
        self.expected_columns: 'Index' = self.rf_judge.feature_names_in_
        
        # Initialize style feature extractor
        self.style_extractor: StyleExtractor = StyleExtractor()
        
        print("Detector is ready and loaded in memory!")

    def _calculate_style(self, text: str) -> Dict[str, Any]:
        """Extract stylometric features from text.
        
        Private method: preprocess text and compute 20+ linguistic metrics.
        
        Parameters
        ----------
        text : str
            Raw input text.
        
        Returns
        -------
        dict
            Stylometric features (punctuation, sentiment, readability, etc.).
        """
        # Normalize text: mask URLs, mentions, numbers
        clean_text: str = self.style_extractor._normalize_text(text)
        # Compute all stylometric features
        return self.style_extractor._extract_metrics(clean_text)

    def analyze(self, text: str) -> Dict[str, Any]:
        """Classify text as fake or real news using Y-architecture ensemble.
        
        Pipeline:
        1. Tokenize text with RoBERTa (max 512 tokens)
        2. Get RoBERTa fake news probability
        3. Extract style features via StyleExtractor
        4. Merge RoBERTa output with style features
        5. Run Random Forest classifier
        6. Return is_fake, confidence, roberta_opinion
        
        Parameters
        ----------
        text : str
            Raw news text or arbitrary paragraph.
        
        Returns
        -------
        dict
            Keys: 'text', 'is_fake_news' (bool), 'rf_confidence' (0-100),
                  'roberta_opinion' (0-100).
        
        Notes
        -----
        Confidence is probability of the predicted class (not always fake).
        Example: if is_fake_news=True, rf_confidence=87% means RF is 87%
        confident it's fake. If is_fake_news=False, rf_confidence=92% means
        RF is 92% confident it's real.
        """
        # Step 1: Tokenize with RoBERTa
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Step 2: Get RoBERTa semantic prediction (no gradient)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        
        # Convert logits to normalized probabilities (label 1 = fake)
        probabilities: 'np.ndarray' = softmax(outputs.logits.numpy()[0])
        roberta_fake_prob: float = probabilities[1]
        
        # Step 3: Extract style features
        style_metrics: Dict[str, Any] = self._calculate_style(text)
        
        # Step 4: Merge RoBERTa probability with style features
        style_metrics['roberta_proba'] = roberta_fake_prob
        df_inference: pd.DataFrame = pd.DataFrame([style_metrics])
        
        # Ensure RF feature columns exist (missing features = 0.0)
        for col in self.expected_columns:
            if col not in df_inference.columns:
                df_inference[col] = 0.0
        
        # Align column order to match RF training
        df_inference = df_inference[self.expected_columns]
        
        # Step 5: Classify with Random Forest
        prediction: int = self.rf_judge.predict(df_inference)[0]
        final_proba: 'np.ndarray' = self.rf_judge.predict_proba(df_inference)[0]
        
        # Step 6: Format output
        is_fake: bool = bool(prediction == 1)
        # Confidence = probability of predicted class (not always fake)
        confidence: float = final_proba[1] * 100 if is_fake else final_proba[0] * 100
        
        return {
            "text": text,
            "is_fake_news": is_fake,
            "rf_confidence": confidence,
            "roberta_opinion": roberta_fake_prob * 100
        }

if __name__ == "__main__":
    # Initialize detector (loads all models into RAM)
    detector: FakeNewsDetector = FakeNewsDetector()
    
    # Interactive mode: test arbitrary inputs
    print("\n Type 'q' to quit.")
    while True:
        # Accept user input (special characters, URLs, mentions, emojis all ok)
        sentence: str = input("\nEnter a sentence to test (#, @, urls or emojis are accepted): ")
        if sentence.lower() in ['q', 'quit']:
            break
        
        # Analyze single sentence through ensemble pipeline
        result: Dict[str, Any] = detector.analyze(sentence)
        
        # Display result with confidence
        if result["is_fake_news"]:
            print(f"FAKE NEWS (Confidence: {result['rf_confidence']:.1f}%)")
            print(f"Input text: {result['text']}")
        else:
            print(f"REAL NEWS (Confidence: {result['rf_confidence']:.1f}%)")
            print(f"Input text: {result['text']}")