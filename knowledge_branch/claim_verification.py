from transformers import pipeline

class ClaimVerifier:

  def __init__ (self):
    model_name = "MoritzLaurer/DeBERTa-v3-base-zeroshot-v1.1-all-33"
    print("Chargement du modèle de vérification")

    self.classifier = pipeline("zero-shot-classification", model=model_name, device=0)


  def verify(self, claim, evidence_text):
    if not evidence_text:
        return "NOT ENOUGH INFO", 0.0

    # On utilise des labels plus explicites pour le modèle
    labels = ["true", "false", "unrelated"]
    
    # On change le template pour être plus direct sur la vérification de fait
    template = f"Based on the Wikipedia text, the statement '{claim}' is {{}}."
    
    result = self.classifier(
        evidence_text, 
        candidate_labels=labels, 
        hypothesis_template=template
    )
    
    # Mapping vers tes labels de sortie
    mapping = {"true": "SUPPORTED", "false": "REFUTED", "unrelated": "NEUTRAL"}
    top_label = mapping[result['labels'][0]]
    top_score = result['scores'][0]
    
    return top_label, top_score