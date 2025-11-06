import streamlit as st
import joblib
import re
import spacy
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from deep_translator import GoogleTranslator
from sklearn.base import BaseEstimator, TransformerMixin

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

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

# Load the trained pipeline
pipeline = joblib.load('clf_pipe.pkl')

# Streamlit interface
st.title("Mobile Review Sentiment Analyzer")
st.write("Enter your mobile product review below:")

review = st.text_area("Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        prediction = pipeline.predict([review])[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write(f"Predicted Sentiment: {sentiment_map.get(prediction, 'Unknown')}")
    else:
        st.warning("Please enter a review to analyze.")
