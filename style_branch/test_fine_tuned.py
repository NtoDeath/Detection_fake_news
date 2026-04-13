"""Interactive testing interface for fine-tuned RoBERTa model.

Loads locally trained model and tokenizer. Provides real-time probability
prediction for arbitrary text input.

Label Encoding
--------------
0 = True news
1 = Fake news

Usage
-----
Run script to test predefined examples, then enter interactive mode.
Type 'q' or 'quit' to exit.

Dependencies
------------
- torch: inference on GPU/CPU
- transformers: tokenizer and model loading
- scipy: probability normalization (softmax)
"""

import torch
import os
import sys
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

print("Loading YOUR local RoBERTa expert...")
model_dir: str = "./roberta_fine_tunned"

# Validate model directory existence before loading
if not os.path.exists(model_dir):
    print(f"CRITICAL ERROR: The directory '{model_dir}' was not found. Have you run the training script yet?")
    sys.exit(1)

# Load tokenizer and model from local checkpoint
tokenizer: 'AutoTokenizer' = AutoTokenizer.from_pretrained(model_dir)
model: 'AutoModelForSequenceClassification' = AutoModelForSequenceClassification.from_pretrained(model_dir)

def analyze_sentence(text: str) -> Dict[str, float]:
    """Predict and display fake news probability for input text.
    
    Parameters
    ----------
    text : str
        Input sentence or paragraph to classify.
    
    Returns
    -------
    dict
        Keys: 'true_proba', 'fake_proba' (float, 0-100 percentage).
    
    Notes
    -----
    Text truncated to 512 tokens. Probabilities computed via softmax
    normalization of model logits.
    """
    # Tokenize input text with truncation to model max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Inference: forward pass without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Normalize logits to probabilities via softmax
    probabilities: 'np.ndarray' = softmax(outputs.logits.numpy()[0])
    
    # Extract class probabilities (label 0=True, 1=Fake)
    true_proba: float = probabilities[0] * 100
    fake_proba: float = probabilities[1] * 100
    
    # Display results
    print(f"\nAnalyzed sentence: '{text}'")
    print(f"TRUE NEWS probability: {true_proba:.2f}%")
    print(f"FAKE NEWS probability: {fake_proba:.2f}%")
    
    return {"true_proba": true_proba, "fake_proba": fake_proba}

if __name__ == "__main__":
    # Demo: test with true news (should have high 'true_proba')
    analyze_sentence("The Federal Reserve announced a 0.5% increase in interest rates today.")

    # Demo: test with clearly fake news (should have high 'fake_proba')
    analyze_sentence("BREAKING: Alien spaceship found hidden under the White House! The government is lying to us!!!")
    
    print("\n" + "="*70)
    print("✅ Model test completed successfully!")
    print("="*70)