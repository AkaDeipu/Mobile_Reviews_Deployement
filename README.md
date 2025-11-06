# 📱 Mobile Reviews Sentiment Analyzer

This project analyzes mobile product reviews and predicts sentiment using a trained ML pipeline. The app is built with Streamlit and features emoji-based sentiment display, a mobile-themed interface, and animated background.

---

## 🚀 Features

- Emoji-based sentiment output (😊 Neutral 😐 Negative 😠)
- Mobile-themed design with image and animation
- Multilingual review support with transliteration
- Deployed via Streamlit Cloud

---

## 🧠 Model Pipeline

The pipeline includes:

- Text preprocessing
- TF-IDF vectorization
- Stacking classifier with multiple base models

Serialized as `clf_pipe.pkl` using `joblib`.

---

## 🛠 Installation (Local)

```bash
git clone https://github.com/akadeipu/mobile_reviews_deployement.git
cd mobile_reviews_deployement
pip install -r requirements.txt
streamlit run app.py
