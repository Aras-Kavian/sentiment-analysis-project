#!/usr/bin/env python3
"""
Simple Streamlit web interface for the sentiment analysis model.

Usage:
streamlit run src/app_streamlit.py
"""
import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path("src/sentiment_model.pkl")
VECTORIZER_PATH = Path("src/vectorizer.pkl")

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="centered")

    st.title("🧠 Sentiment Analysis")
    st.write("Type a sentence and get the predicted sentiment (positive / negative).")

    model, vectorizer = load_model_and_vectorizer()

    text = st.text_area("✍️ Enter text to analyze", height=150)

    if st.button("Analyze"):
        if text.strip():
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            label = "Positive 😊" if pred == 1 else "Negative 😠"
            st.success(f"**Prediction:** {label}")
        else:
            st.warning("Please enter some text.")

    st.markdown("---")
    st.caption("Built with Streamlit · Logistic Regression · Scikit-learn")

if __name__ == "__main__":
    main()