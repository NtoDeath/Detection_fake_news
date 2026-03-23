"""Claim verification via natural language inference (entailment).

Verifies factual claims by comparing evidence text against claim statement.
Uses zero-shot entailment classification to determine support status.

Entailment Pyramid
------------------
SUPPORTED: Evidence clearly entails the claim (confidence >= 70%)
REFUTED: Evidence contradicts the claim (not_entailment detected)
UNVERIFIED: Evidence discusses topic but doesn't confirm action (low confidence)
NOT ENOUGH INFO: No evidence provided

Model
-----
DeBERTa-v3-base zero-shot NLI (Natural Language Inference).
Task: [evidence_text] + [SEP] + [claim] -> entailment | not_entailment

Dependencies
------------
- transformers: HuggingFace pipeline for zero-shot classification
"""

from typing import Tuple, Dict, Any, Optional
from transformers import pipeline


class ClaimVerifier:
    """Verify claims using natural language inference on evidence.
    
    Attributes
    ----------
    classifier : transformers.Pipeline
        DeBERTa zero-shot entailment model.
    
    Notes
    -----
    Evidence text truncated to 1000 characters for computational efficiency.
    Confidence scoring: 0-1 (converted to percentage in output).
    """

    def __init__(self) -> None:
        """Initialize DeBERTa entailment classifier.
        
        Loads zero-shot NLI model from HuggingFace.
        Requires GPU (device=0) for fast inference.
        
        Notes
        -----
        First initialization downloads ~500MB model.
        Subsequent runs use cached checkpoint.
        """
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-zeroshot-v1.1-all-33"
        print("Chargement du modèle de vérification")

        # Load zero-shot classification pipeline on GPU device 0
        self.classifier: 'transformers.Pipeline' = pipeline("text-classification", model=model_name, device=0)


    def verify(self, claim: str, evidence_text: Optional[str]) -> Tuple[str, float]:
        """Verify a claim against evidence using entailment.
        
        Pipeline:
        1. Check if evidence provided (early exit if empty)
        2. Format input: [evidence_text][:1000] [SEP] [claim]
        3. Run DeBERTa entailment classifier
        4. Map predictions to verdict (SUPPORTED/REFUTED/UNVERIFIED)
        5. Return verdict and confidence score
        
        Parameters
        ----------
        claim : str
            Factual claim to verify (e.g., "The government is hiding aliens").
        evidence_text : str or None
            Retrieved evidence from knowledge base.
        
        Returns
        -------
        tuple of (str, float)
            - verdict: "SUPPORTED", "REFUTED/UNVERIFIED", "NOT ENOUGH INFO", or error msg
            - score: confidence (0.0 to 1.0)
        
        Notes
        -----
        Evidence truncated to 1000 chars to stay within DeBERTa context limit.
        Thresholds:
        - entailment + score >= 0.70 -> SUPPORTED
        - not_entailment -> REFUTED/UNVERIFIED (evidence contradicts claim)
        - entailment + score <= 0.85 -> UNVERIFIED (weak signal)
        
        Exception Handling
        ------------------
        Returns ("ERROR: ...", 0.0) on model inference failures.
        """
        # Handle case: no evidence available for comparison
        if not evidence_text:
            return "NOT ENOUGH INFO", 0.0
        
        # Format input for entailment: evidence [SEP] claim
        # Truncate evidence to 1000 chars for computational efficiency
        text_input: str = f"{evidence_text[:1000]} [SEP] {claim}"

        try:
            # Run zero-shot classification (produces entailment/contradiction score)
            result: Dict[str, Any] = self.classifier(text_input)[0]

            label_detected: str = result['label']
            score: float = result['score']

            # Entailment classification logic
            # entailment = evidence supports claim
            # not_entailment = evidence contradicts or is silent on claim
            if label_detected == "entailment" and score >= 0.70:
                # High confidence: evidence clearly supports claim
                return "SUPPORTED", score
            elif label_detected == "not_entailment" or (label_detected == "entailment" and score <= 0.85):
                # Low confidence OR contradiction: source discusses topic but doesn't confirm action
                # Example: "Area 51 is a military base" vs "Area 51 is hiding aliens"
                return "REFUTED / UNVERIFIED", score
            else:
                # Intermediate confidence: ambiguous relationship
                return "NEUTRAL", score

        except Exception as e:
            # Inference error: model failure or input processing issue
            return f"ERROR: {str(e)}", 0.0