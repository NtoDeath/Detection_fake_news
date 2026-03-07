class EvidenceRetriever:
    def __init__(self, model):
        self.model = model

    def retrieve_evidence(self, claim):
        # Implement the evidence retrieval logic using the model
        return self.model.retrieve(claim)