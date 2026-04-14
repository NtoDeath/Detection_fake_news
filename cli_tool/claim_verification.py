"""
Claim verification using DeBERTa NLI model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple
import torch.nn.functional as F


class ClaimVerifier:
    """Verify claims against evidence using DeBERTa NLI"""
    
    def __init__(self):
        """Initialize the DeBERTa NLI model"""
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        
        # Choose device
        device = 0 if torch.cuda.is_available() else -1
        self.device = torch.device("cuda" if device >= 0 else "cpu")
        
        # Load tokenizer and model directly for more control
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.tokenizer = None
            self.model = None
        
        # Label mapping: the model outputs [contradiction, neutral, entailment]
        self.id_to_label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def verify(self, claim: str, evidence_text: str) -> Tuple[str, float]:
        """
        Verify a claim against evidence.
        
        Args:
            claim: The claim to verify
            evidence_text: The evidence to check against
        
        Returns:
            Tuple of (verdict, confidence)
            Verdicts: "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"
        """
        
        if not evidence_text or len(evidence_text) < 20:
            return "NOT_ENOUGH_INFO", 0.0

        if self.model is None or self.tokenizer is None:
            return "ERROR: Model not loaded", 0.0

        # Add context that this is factual evidence (improves contradiction detection)
        evidence_context = f"Factual context: {evidence_text[:1500]}"
        text_input = f"{evidence_context} [SEP] {claim}"

        try:
            # Tokenize
            inputs = self.tokenizer(
                text_input,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)[0]
            
            # Extract scores for each label: [contradiction, neutral, entailment]
            contradiction_score = probs[0].item()
            neutral_score = probs[1].item()
            entailment_score = probs[2].item()
            
            # Decision logic:
            # 1. If entailment is highest and >0.5 → SUPPORTED
            # 2. If contradiction is highest and >0.4 → REFUTED
            # 3. Otherwise → NOT_ENOUGH_INFO
            
            # Find top score
            if entailment_score > 0.5 and entailment_score >= contradiction_score:
                return "SUPPORTED", entailment_score
            elif contradiction_score > 0.4:
                return "REFUTED", contradiction_score
            elif entailment_score > contradiction_score and entailment_score > 0.3:
                return "SUPPORTED", entailment_score
            else:
                return "NOT_ENOUGH_INFO", max(neutral_score, min(contradiction_score, entailment_score))

        except Exception as e:
            return f"ERROR: {str(e)}", 0.0
