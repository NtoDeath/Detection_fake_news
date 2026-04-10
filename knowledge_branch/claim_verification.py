from transformers import pipeline

class ClaimVerifier:
    def __init__(self):
        # Utilisation du modèle DeBERTa spécialisé
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.classifier = pipeline("text-classification", model=self.model_name, device=0)

    def verify(self, claim, evidence_text):
        if not evidence_text or len(evidence_text) < 20:
            return "NOT ENOUGH INFO", 0.0

        # Format optimal pour DeBERTa : la preuve d'abord, puis le claim
        text_input = f"{evidence_text[:1500]} [SEP] {claim}"

        try:
            result = self.classifier(text_input)[0]
            label = result['label'].lower()
            score = result['score']

            # Les modèles MNLI/FEVER utilisent souvent ces labels :
            if "entailment" in label:
                return "SUPPORTED", score
            elif "contradiction" in label:
                return "REFUTED", score
            else:
                return "NEUTRAL / NOT ENOUGH INFO", score

        except Exception as e:
            return f"ERROR: {str(e)}", 0.0