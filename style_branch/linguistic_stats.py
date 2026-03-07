class LinguisticStats:
    def __init__(self, text):
        self.text = text

    def word_count(self):
        return len(self.text.split())

    