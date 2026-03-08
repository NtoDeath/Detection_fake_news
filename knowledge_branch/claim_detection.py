class ClaimDetector:
    def __init__(self, model):
        self.model = model

    def detect_claims(self, text):
        # Implement the claim detection logic using the model
        return self.model.detect(text)