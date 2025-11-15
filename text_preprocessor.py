# text_preprocessor.py

import re
import spacy
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from deep_translator import GoogleTranslator
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_sm")

negations = {"no", "not", "nor", "never", "nothing", "none", "n’t"}
for word in negations:
    if word in nlp.Defaults.stop_words:
        nlp.Defaults.stop_words.remove(word)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._preprocess(text) for text in X]

    def _preprocess(self, text):
        try:
            lang = detect(text)
            if lang != 'en':
                devan = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                text = GoogleTranslator(source='hi', target='en').translate(devan)
        except Exception:
            pass

        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
        text = re.sub(r"([(?.!,@¿)])", " ", text)
        text = re.sub(r'["\']+', " ", text)
        text = re.sub(r"http\S+|https\S+|www\S+", '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()

        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token) > 2]
        return " ".join(tokens)
