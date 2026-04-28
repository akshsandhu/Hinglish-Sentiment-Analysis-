"""
predict.py
──────────
Load fine-tuned IndicBERT and make sentiment predictions.
Used by app.py for the Streamlit UI.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_hinglish_text

MODEL_PATH = "models/indicbert-hinglish"

ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
EMOJIS   = {0: "😊",       1: "😞",       2: "😐"}


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Train the model in Colab first, then download it here."
        )
    print(f"📦 Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"   Device: {device} ✓")
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model, max_length: int = 128) -> dict:
    clean = clean_hinglish_text(text)

    if len(clean) == 0:
        return {
            "label": "Neutral", "label_id": 2,
            "confidence": 0.5, "probs": [0.33, 0.33, 0.34],
            "emoji": "😐", "clean_text": clean,
        }

    device = next(model.parameters()).device
    inputs = tokenizer(
        clean, return_tensors="pt",
        truncation=True, padding="max_length", max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    probs_list = probs.cpu().numpy().tolist()
    pred_id    = int(probs.argmax())
    confidence = float(probs[pred_id])

    return {
        "label"      : ID2LABEL[pred_id],
        "label_id"   : pred_id,
        "confidence" : confidence,
        "probs"      : probs_list,
        "emoji"      : EMOJIS[pred_id],
        "clean_text" : clean,
    }


if __name__ == "__main__":
    try:
        tokenizer, model = load_model()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        exit()

    test_texts = [
        "Bhut accha product hai! Delivery bhi fast thi.",
        "Bekar quality hai, bilkul waste of money.",
        "Theek thaak hai. Not bad but could be better.",
        "Zabardast! Best phone in this price range.",
    ]

    print(f"\n🧪 Testing {len(test_texts)} texts:\n")
    for text in test_texts:
        r = predict_sentiment(text, tokenizer, model)
        print(f"📝 {text}")
        print(f"   {r['emoji']} {r['label']:10s} | {r['confidence']*100:.1f}%\n")