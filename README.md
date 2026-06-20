# Hinglish Sentiment Analysis

Fine-tuned IndicBERT on 14K+ Hinglish tweets from SemEval 2020 for multilingual sentiment classification. Built an end-to-end NLP pipeline covering data parsing, preprocessing, transformer fine-tuning, inference, and deployment through an interactive Streamlit application.

## 🚀 Project Overview

This project classifies Hinglish (Hindi-English code-mixed) text into:

* Positive 😊
* Negative 😞
* Neutral 😐

The system is designed to handle real-world social media text containing:

* Hindi written in Roman script
* English words
* Mixed-language sentences
* Informal slang and abbreviations

## ✨ Key Features

* Fine-tuned IndicBERT for Hinglish sentiment analysis
* Trained on 14K+ SemEval 2020 Hinglish tweets
* Custom text normalization and slang handling
* GPU-based transformer fine-tuning
* Real-time sentiment prediction
* Confidence score visualization
* Aspect detection (Quality, Delivery, Price, Service)
* Interactive Streamlit dashboard
* Hindi + English + Hinglish language support

## 🏗️ Architecture

Dataset (SemEval 2020)
↓
CoNLL Parser
↓
Text Cleaning & Normalization
↓
Train / Validation / Test Split
↓
IndicBERT Fine-Tuning
↓
Sentiment Prediction Engine
↓
Streamlit Web Application

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers
* IndicBERT
* PyTorch
* Scikit-Learn
* Pandas
* NumPy
* Streamlit

## 📊 Model Capabilities

The model performs sentiment classification on code-mixed Hindi-English text and generates:

* Predicted sentiment label
* Confidence score
* Probability distribution
* Cleaned text preview
* Aspect-based keyword detection

## 📈 Results

* Dataset: 14K+ Hinglish Tweets
* Model: IndicBERT
* Accuracy: ~88%
* Classes: Positive, Negative, Neutral

## 🎯 Example

Input:

"Bhut accha product hai! Delivery bhi fast thi."

Output:

Sentiment: Positive 😊
Confidence: 94%

## 💻 Run Locally

```bash
git clone https://github.com/yourusername/Hinglish-Sentiment-Analysis.git
cd Hinglish-Sentiment-Analysis

pip install -r requirements.txt

streamlit run app.py
```

## 📷 Demo

The Streamlit application provides:

* Real-time sentiment analysis
* Confidence visualization
* Interactive examples
* Aspect detection
* Clean text inspection

## 📚 Dataset

SemEval 2020 Task 9:
Sentiment Analysis for Code-Mixed Social Media Text

## 👨‍💻 Author

Aksh Sandhu

LinkedIn: www.linkedin.com/in/aksh-sandhu-8282a63b7

GitHub: https://github.com/akshsandhu
