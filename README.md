# 🧠 Sentiment Analysis Project

This is a simple **Sentiment Analysis** project built with Python.  
The goal is to classify text as **Positive**, **Negative**, or **Neutral** using a basic machine learning model.

## 🚀 Features
- Preprocessing of text (tokenization, cleaning, stopword removal)
- Simple ML model for classification (Logistic Regression / Naive Bayes)
- Easy-to-use interface for testing new sentences

## 🛠 Tech Stack
- Python 3.x
- scikit-learn
- pandas / numpy
- nltk

## 📂 Project Structure
sentiment-analysis-project/
- data/______________# Dataset files
- notebooks/_________# Jupyter notebooks for experimentation
- src/_______________# Source code
- README.md
- requirements.txt

## 📊 Example Output
Input: “I love this product!”
Prediction: Positive ✅

## 📊 Model Training
We trained a simple Logistic Regression model on the IMDb dataset to classify text as positive or negative.  
The model achieved an accuracy of around **88%** on the test set.  
Model and vectorizer are saved for later use in a simple CLI/Web interface.

## 🚀 Usage - Command Line Interface

After training the model, you can run an interactive CLI tool:

'''bash
python src/app.py
Then type any sentence to analyze its sentiment.
Type exit or quit to close the program.

Example:
Enter text: This movie was fantastic!
Prediction: POSITIVE'''

## 📝 Next Steps 
- [ ] Build simple CLI / Streamlit UI  
- [ ] Deploy model (Optional)

---

👨‍💻 **Author:** [Aras Kavyani]  
🔗 [LinkedIn](#www.linkedin.com/in/aras-kavyani) | [LaborX Profile](#www.laborx.com/customers/users/id409982?ref=409982) | [CryptoTask Profile](#www.cryptotask.org/en/freelancers/aras-kavyan/46480)
