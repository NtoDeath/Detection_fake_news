import torch
import joblib
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from style_extractor import StyleExtractor

class FakeNewsDetector:
    """
    Production object handling the Y-architecture: 
    Style Extractor + RoBERTa -> Random Forest
    """
    
    def __init__(self, roberta_path="./roberta_fine_tunned", rf_path="best_model_random_forest.pkl"):
        print("Initializing Detector (Loading into RAM...)")
        
        if not os.path.exists(roberta_path) or not os.path.exists(rf_path):
            raise FileNotFoundError(
                f"Missing model files! Ensure both '{roberta_path}' and '{rf_path}' exist."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_path)
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path)
        
        self.rf_judge = joblib.load(rf_path)
        self.expected_columns = self.rf_judge.feature_names_in_
        
        print("Detector is ready and loaded in memory!")

    def _calculate_style(self, text):
        """Private method to compute style metrics"""
        extractor = StyleExtractor()
        clean_text = extractor._normalize_text(text)
        return extractor._extract_metrics(clean_text)

    def analyze(self, text):
        """Public method to be called by your future website"""
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        
        probabilities = softmax(outputs.logits.numpy()[0])
        roberta_fake_prob = probabilities[1]
        
        style_metrics = self._calculate_style(text)
        
        style_metrics['roberta_proba'] = roberta_fake_prob
        df_inference = pd.DataFrame([style_metrics])
        
        for col in self.expected_columns:
            if col not in df_inference.columns:
                df_inference[col] = 0.0
                
        df_inference = df_inference[self.expected_columns]
        
        prediction = self.rf_judge.predict(df_inference)[0]
        final_proba = self.rf_judge.predict_proba(df_inference)[0]
        
        is_fake = bool(prediction == 1)
        confidence = final_proba[1] * 100 if is_fake else final_proba[0] * 100
        
        return {
            "text": text,
            "is_fake_news": is_fake,
            "rf_confidence": confidence,
            "roberta_opinion": roberta_fake_prob * 100
        }

if __name__ == "__main__":
    detector = FakeNewsDetector()
    
    print("\n Type 'q' to quit.")
    while True:
        sentence = input("\nEnter a sentence to test (#, @, urls or emojis are accepted): ")
        if sentence.lower() in ['q', 'quit']:
            break
            
        result = detector.analyze(sentence)
        
        if result["is_fake_news"]:
            print(f"FAKE NEWS (Confidence: {result['rf_confidence']:.1f}%)")
            print(f"Input text: {result['text']}")
        else:
            print(f"REAL NEWS (Confidence: {result['rf_confidence']:.1f}%)")
            print(f"Input text: {result['text']}")