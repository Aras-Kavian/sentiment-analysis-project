#!/usr/bin/env python3
"""
Streamlit web interface for the sentiment analysis model with enhanced UI.

Usage:
streamlit run src/app_streamlit.py
"""
import streamlit as st
import joblib
from pathlib import Path

MODEL_PATH = Path("src/sentiment_model.pkl")
VECTORIZER_PATH = Path("src/vectorizer.pkl")

# -------------------------------------------------------
# Load model and vectorizer
# -------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# -------------------------------------------------------
# Main app
# -------------------------------------------------------
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="🧠", layout="centered")

    st.title("🧠 Sentiment Analysis")
    st.write("Type a sentence and get the predicted sentiment (positive / negative).")

    model, vectorizer = load_model_and_vectorizer()

    text = st.text_area("✍️ Enter text to analyze", height=150)

    if st.button("Analyze"):
        if text.strip():
            # Transform text and predict
            X = vectorizer.transform([text])
            pred_prob = model.predict_proba(X)[0]  # get probabilities for both classes
            pred_class = model.predict(X)[0]

            # Class labels with emojis
            CLASSES = ["Negative 😠", "Positive 😊"]
            COLORS = ["#FF6F61", "#4CAF50"]  # red for negative, green for positive

            # Display predicted class
            st.markdown(f"### ✅ Predicted Sentiment: **{CLASSES[pred_class]}**")

            # Display confidence bars
            st.write("📊 Confidence Scores:")
            for i, (label, prob, color) in enumerate(zip(CLASSES, pred_prob, COLORS)):
                percent = prob * 100
                st.markdown(
                    f"""
                    <div style='display:flex; align-items:center; margin-bottom:10px;'>
                        <div style='font-size:24px; width:80px;'>{label.split()[1]}</div>
                        <div style='flex:1; background-color:#e0e0e0; border-radius:5px; margin-left:10px;'>
                            <div style='width:{percent}%; background-color:{color}; padding:5px 0; border-radius:5px; text-align:center; color:white; font-weight:bold;'>
                                {percent:.2f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("---")
    st.caption("Built with Streamlit · Logistic Regression · Scikit-learn")

if __name__ == "__main__":
    main()
