import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

class TextPreprocessor(
    BaseEstimator,
    TransformerMixin
):
    def __init__(self):
        self.port_stem = PorterStemmer()
        negation_words = {
            'not',
            'no',
            'nor',
            'never'
        }
        self.stop_words = (
            set(stopwords.words('english'))
            - negation_words
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_text = []
        for content in X:
            content = re.sub(
                '[^a-zA-Z]',
                ' ',
                content
            )
            content = content.lower()
            words = content.split()
            words = [
                self.port_stem.stem(word)
                for word in words
                if word not in self.stop_words
            ]
            processed_text.append(
                ' '.join(words)
            )

        return processed_text
