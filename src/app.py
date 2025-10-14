#!/usr/bin/env python3
"""
Simple CLI interface for the sentiment analysis model.

Usage:
python src/app.py
"""
import joblib
from pathlib import Path

MODEL_PATH = Path("src/sentiment_model.pkl")
VECTORIZER_PATH = Path("src/vectorizer.pkl")

def load_model_and_vectorizer():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model or vectorizer file not found. Please run train.py first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "positive" if pred == 1 else "negative"

def main():
    print("="*60)
    print("🧠 Sentiment Analysis CLI")
    print("Type a sentence to analyze its sentiment.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60)

    model, vectorizer = load_model_and_vectorizer()

    while True:
        user_input = input("\nEnter text: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\n👋 Exiting. Goodbye!")
            break
        if not user_input:
            print("⚠️ Please enter some text.")
            continue
        sentiment = predict_sentiment(model, vectorizer, user_input)
        print(f"Prediction: {sentiment.upper()}")

if __name__ == "__main__":
    main()