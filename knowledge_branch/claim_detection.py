from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas

class ClaimDetector:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        model_name = "tals/albert-base-v2-checkworthiness"
        self.classifier = pipeline("text-classification", model=model_name)

    def process_text(self, text):
        #1. segmentation avec spacy

        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]

        #2. classification avec le modèle BERT checkworthiness

        results = self.classifier(sentences)

        claims = []
        for sentence, result in zip(sentences, results):
            if result['label'] == 'LABEL_1':  # Assuming LABEL_1 indicates checkworthy
                claims.append(sentence.strip())
        return claims

    def load_data(self, file_path):
        df = pandas.read_json(file_path, lines=True)
        return df['text'].tolist()  # Assuming the text column is named 'text'