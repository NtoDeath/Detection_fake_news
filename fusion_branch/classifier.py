class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        # Implement the prediction logic using the model
        return self.model.predict(input_data)