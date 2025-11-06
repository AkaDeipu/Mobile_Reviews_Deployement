import streamlit as st
import joblib
import re
import spacy
from text_preprocessor import TextPreprocessor
import base64
# Load the trained pipeline
pipeline = joblib.load('clf_pipe.pkl')

# Custom CSS for animated background 
def set_background():
    st.markdown("""
        <style>
        body {
            background: linear-gradient(-45deg, #ffecd2, #fcb69f, #ff9a9e, #fad0c4);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .mobile-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 120px;
        }
        </style>
    """, unsafe_allow_html=True)

# Emoji map
sentiment_emojis = {
    0: "😠 Negative",
    1: "😐 Neutral",
    2: "😊 Positive"
}

# Streamlit interface
set_background()
st.title("Mobile Review Sentiment Analyzer")
st.write("Enter your mobile product review below:")

review = st.text_area("Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        proba = pipeline.predict_proba([review])[0]
        prediction = proba.argmax()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write(f"Predicted Sentiment: {sentiment_map.get(prediction, 'Unknown')}")
        st.markdown(f"## Sentiment: {sentiment_emojis.get(prediction, 'Unknown')}")
        st.write(f"Confidence Scores:")
        st.write({
            "Negative": round(proba[0], 3),
            "Neutral": round(proba[1], 3),
            "Positive": round(proba[2], 3)})
    else:
        st.warning("Please enter a review to analyze.")
