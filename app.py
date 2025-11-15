import streamlit as st
import pandas as pd
import joblib
import re
import spacy
from text_preprocessor import TextPreprocessor
nlp = spacy.load("en_core_web_sm")

#importing the pipeline for the new review analysis
#My pipeline has the text translation, text preprocessing, vectorizer, model
pipeline = joblib.load('clf_pipe.pkl')

# Emoji map for the output
sentiment_emojis = {
    0: "😠 Negative",
    1: "😐 Neutral",
    2: "😊 Positive"
}

# Creating a simple interface for better visualization
st.title("Mobile Review Sentiment Analyzer")
st.write("Enter your mobile product review below:")

review = st.text_area("Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        proba = pipeline.predict_proba([review])[0]
        prediction = proba.argmax()
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        #st.write(f"Predicted Sentiment: {sentiment_map.get(prediction, 'Unknown')}")
        st.markdown(f"## Predicted Sentiment: {sentiment_emojis.get(prediction, 'Unknown')}")
        #st.write(f"## Confidence Scores:")
        #st.write({
            #"Negative": round(proba[0], 3),
            #"Neutral": round(proba[1], 3),
            #"Positive": round(proba[2], 3)})

        st.subheader("Confidence Scores")
        proba_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability": proba
        })
        st.bar_chart(proba_df.set_index("Sentiment"))

        doc = nlp(review.lower())
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space]

        
        positive_words_list = ["good", "great", "excellent", "amazing", "love", "fantastic", "awesome", "wonderful"]
        negative_words_list = ["bad", "poor", "terrible", "hate", "awful", "worst", "disappointing", "horrible"]

        positive_words = [w for w in tokens if w in positive_words_list]
        negative_words = [w for w in tokens if w in negative_words_list]
        neutral_words = [w for w in tokens if w not in positive_words_list and w not in negative_words_list]

        st.subheader("Words by Sentiment")
        st.write("**Positive Words:**", ", ".join(positive_words) if positive_words else "None")
        st.write("**Negative Words:**", ", ".join(negative_words) if negative_words else "None")
        st.write("**Neutral Words:**", ", ".join(neutral_words) if neutral_words else "None")

    else:
        st.warning("Please enter a review to analyze.")
