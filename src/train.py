#!/usr/bin/env python3
"""
Train a simple sentiment analysis model and save model + vectorizer.

Usage:
python src/train.py --data_path data/IMDB\ Dataset.csv --model_out src/sentiment_model.pkl --vector_out src/vectorizer.pkl
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

def main(args):
    data_path = Path(args.data_path)
    model_out = Path(args.model_out)
    vector_out = Path(args.vector_out)
    metrics_out = Path(args.metrics_out)

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    # Expect columns 'review' and 'sentiment'
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain columns named 'review' and 'sentiment'")

    X = df['review'].astype(str)
    y = df['sentiment'].str.lower().map({'positive': 1, 'negative': 0})
    if y.isnull().any():
        raise ValueError("Found unknown labels in 'sentiment' column. Expected 'positive'/'negative'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print("Vectorizing text...")
    vectorizer = CountVectorizer(stop_words='english', max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Ensure output folder exists
    model_out.parent.mkdir(parents=True, exist_ok=True)
    vector_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)
    joblib.dump(vectorizer, vector_out)
    print(f"Saved model -> {model_out}")
    print(f"Saved vectorizer -> {vector_out}")

    # save metrics
    metrics = {'accuracy': acc, 'report': report}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open('w', encoding='utf8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics -> {metrics_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/IMDB Dataset.csv', help='Path to CSV dataset')
    p.add_argument('--model_out', default='src/sentiment_model.pkl', help='Where to save model')
    p.add_argument('--vector_out', default='src/vectorizer.pkl', help='Where to save vectorizer')
    p.add_argument('--metrics_out', default='src/metrics.json', help='Where to save metrics JSON')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--max_features', type=int, default=5000)
    args = p.parse_args()
    main(args)