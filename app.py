import streamlit as st
import joblib
import re
import spacy
from text_preprocessor import TextPreprocessor
# Load the trained pipeline
pipeline = joblib.load('clf_pipe.pkl')

# Streamlit interface
st.title("Mobile Review Sentiment Analyzer")
st.write("Enter your mobile product review below:")

review = st.text_area("Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        proba = pipeline.predict_proba([review])[0]
        prediction = proba.argmax()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write(f"Predicted Sentiment: {sentiment_map.get(prediction, 'Unknown')}")
        st.write(f"Confidence Scores:")
        st.write({"Negative": round(proba[0], 3),"Neutral": round(proba[1], 3),"Positive": round(proba[2], 3)})
    else:
        st.warning("Please enter a review to analyze.")
