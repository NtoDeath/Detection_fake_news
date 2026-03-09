import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

print("Chargement de TON expert RoBERTa local...")
dossier = "./roberta_fine_tunned"

tokenizer = AutoTokenizer.from_pretrained(dossier)
modele = AutoModelForSequenceClassification.from_pretrained(dossier)

def analyser_phrase(texte):
    inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = modele(**inputs)
    
    probabilites = softmax(outputs.logits.numpy()[0])
    
    # Le modèle a été entraîné avec 0 = Vrai, 1 = Faux
    proba_vrai = probabilites[0] * 100
    proba_faux = probabilites[1] * 100
    
    print(f"Phrase analysée : '{texte}'")
    print(f"Probabilité VRAIE INFO : {proba_vrai:.2f}%")
    print(f"Probabilité FAKE NEWS  : {proba_faux:.2f}%")

# Teste avec une vraie info
analyser_phrase("The Federal Reserve announced a 0.5% increase in interest rates today.")

# Teste avec une fake news
analyser_phrase("BREAKING: Alien spaceship found hidden under the White House! The government is lying to us!!!")