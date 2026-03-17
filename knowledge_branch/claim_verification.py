from transformers import pipeline

class ClaimVerifier:

  def __init__ (self):
    model_name = "MoritzLaurer/DeBERTa-v3-base-zeroshot-v1.1-all-33"
    print("Chargement du modèle de vérification")

    self.classifier = pipeline("text-classification", model=model_name, device=0)


  def verify(self, claim, evidence_text):
    if not evidence_text:
        return "NOT ENOUGH INFO", 0.0
    text_input = f"{evidence_text[:1000]} [SEP] {claim}"

    try:

      result = self.classifier(text_input)[0]

      label_detected = result['label']
      score = result['score']

      mapping = {
                "entailment": "SUPPORTED",
                "not_entailment": "REFUTED", # Ou "NEUTRAL/REFUTED"
                "LABEL_0": "SUPPORTED",
                "LABEL_1": "REFUTED"
            }
      if label_detected == "entailment" and score >= 0.70:
        #return mapping[label_detected], score
        return "SUPPORTED", score
      elif label_detected == "not_entailment" or (label_detected == "entailment" and score <= 0.85):
            # Si le score est faible, c'est probablement que la source parle du sujet
            # (Area 51) sans confirmer l'action (hiding aliens)
            return "REFUTED / UNVERIFIED", score
      else:
            return "NEUTRAL", score

    except Exception as e:
      return f"ERROR: {str(e)}", 0.0