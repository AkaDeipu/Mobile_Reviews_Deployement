import streamlit as st
import joblib
import pandas as pd
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import spacy
import scipy

# Loading my pipeline
pipeline = joblib.load('clf_pipe.pkl')

# Emoji map
sentiment_emojis = {
    0: "😠 Negative",
    1: "😐 Neutral",
    2: "😊 Positive"
}

# App layout
st.title("Mobile Review Sentiment Analyzer")
st.markdown("### Enter your mobile product review below:")

review = st.text_area("Your Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        prediction = pipeline.predict([review])[0]
        st.markdown(f"## Sentiment: {sentiment_emojis.get(prediction, '🤔 Unknown')}")
    else:
        st.warning("Please enter a review to analyze.")
