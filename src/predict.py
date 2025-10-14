#!/usr/bin/env python3
"""
Predict sentiment for a single sentence or a file of sentences.

Usage examples:
python src/predict.py --model src/sentiment_model.pkl --vectorizer src/vectorizer.pkl --text "I love this movie!"
python src/predict.py --model src/sentiment_model.pkl --vectorizer src/vectorizer.pkl --input_file data/sample_texts.csv --output_file outputs/preds.csv
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd

def predict_texts(model_path, vectorizer_path, texts):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = ['negative', 'positive']
    return [labels[int(p)] for p in preds]

def main(args):
    model_path = Path(args.model)
    vectorizer_path = Path(args.vectorizer)

    if args.text:
        texts = [args.text]
        preds = predict_texts(model_path, vectorizer_path, texts)
        for t, p in zip(texts, preds):
            print(f"Input: {t}\nPrediction: {p}\n")
    elif args.input_file:
        # try read csv or plain text
        inp = Path(args.input_file)
        if inp.suffix.lower() in {'.csv'}:
            df = pd.read_csv(inp)
            # assume a column named 'text' or first column
            if 'text' in df.columns:
                texts = df['text'].astype(str).tolist()
            else:
                texts = df.iloc[:,0].astype(str).tolist()
        else:
            # plain txt, one line per text
            with inp.open('r', encoding='utf8') as f:
                texts = [line.strip() for line in f if line.strip()]
        preds = predict_texts(model_path, vectorizer_path, texts)
        out = Path(args.output_file or 'outputs/predictions.csv')
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'text': texts, 'prediction': preds}).to_csv(out, index=False)
        print(f"Saved predictions to {out}")
    else:
        print("Provide --text or --input_file")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='src/sentiment_model.pkl')
    p.add_argument('--vectorizer', default='src/vectorizer.pkl')
    p.add_argument('--text', type=str, help='Single text to predict')
    p.add_argument('--input_file', type=str, help='CSV or TXT file with texts')
    p.add_argument('--output_file', type=str, help='Output CSV path')
    args = p.parse_args()
    main(args)