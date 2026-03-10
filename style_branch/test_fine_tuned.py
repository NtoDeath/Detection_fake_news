import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

print("Loading YOUR local RoBERTa expert...")
model_dir = "./roberta_fine_tunned"

if not os.path.exists(model_dir):
    print(f"CRITICAL ERROR: The directory '{model_dir}' was not found. Have you run the training script yet?")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

def analyze_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = softmax(outputs.logits.numpy()[0])
    
    # The model was trained with 0 = True, 1 = Fake
    true_proba = probabilities[0] * 100
    fake_proba = probabilities[1] * 100
    
    print(f"\nAnalyzed sentence: '{text}'")
    print(f"TRUE NEWS probability: {true_proba:.2f}%")
    print(f"FAKE NEWS probability: {fake_proba:.2f}%")
    
    return {"true_proba": true_proba, "fake_proba": fake_proba}

if __name__ == "__main__":
    # Test with real news
    analyze_sentence("The Federal Reserve announced a 0.5% increase in interest rates today.")

    # Test with fake news
    analyze_sentence("BREAKING: Alien spaceship found hidden under the White House! The government is lying to us!!!")
    
    while True:
        user_input = input("\nEnter a sentence to test (or 'q' to quit): ")
        if user_input.lower() in ['q', 'quit']:
            break
        analyze_sentence(user_input)