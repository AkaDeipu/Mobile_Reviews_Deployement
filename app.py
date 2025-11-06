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

# Display mobile image
#def show_mobile_image():
    #with open("mobile.png", "rb") as img_file:
        #b64_img = base64.b64encode(img_file.read()).decode()
        #st.markdown(f'<img src="data:image/png;base64,{b64_img}" class="mobile-img">', unsafe_allow_html=True)

# Emoji map
sentiment_emojis = {
    0: "😠 Negative",
    1: "😐 Neutral",
    2: "😊 Positive"
}

# App layout
#set_background()
st.title("📱 Mobile Review Sentiment Analyzer")
#show_mobile_image()
st.markdown("### Enter your mobile product review below:")

review = st.text_area("✍️ Your Review", height=150)

if st.button("🔍 Analyze Sentiment"):
    if review.strip():
        prediction = pipeline.predict([review])[0]
        st.markdown(f"## Sentiment: {sentiment_emojis.get(prediction, '🤔 Unknown')}")
    else:
        st.warning("Please enter a review to analyze.")
