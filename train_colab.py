"""
train_colab.py
──────────────
Fine-tune IndicBERT for Hinglish sentiment analysis.
RUN THIS IN GOOGLE COLAB (needs free T4 GPU).

Steps:
1. Open colab.research.google.com
2. Runtime → Change runtime type → T4 GPU
3. Upload train.csv, val.csv, test.csv from your data/ folder
4. Paste this entire file into a Colab cell and run it
5. Model saves to Google Drive automatically
"""

# ── Install packages (run this cell first in Colab) ──────
# !pip install transformers datasets torch accelerate sentencepiece -q

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import os

print(f"✅ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_NAME    = "ai4bharat/indic-bert"
NUM_LABELS    = 3
MAX_LENGTH    = 128
BATCH_SIZE    = 16
EPOCHS        = 5
LEARNING_RATE = 2e-5
OUTPUT_DIR    = "./results"
SAVE_PATH     = "./models/indicbert-hinglish"

ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}
LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}


def load_data():
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv("train.csv")
    val_df   = pd.read_csv("val.csv")
    test_df  = pd.read_csv("test.csv")

    # Ensure clean_text column exists
    for df in [train_df, val_df, test_df]:
        if "clean_text" not in df.columns:
            df["clean_text"] = df["text"]

    print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def tokenize_data(train_df, val_df, test_df):
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]])
    val_dataset   = Dataset.from_pandas(val_df[["clean_text", "label"]])
    test_dataset  = Dataset.from_pandas(test_df[["clean_text", "label"]])

    print("   Tokenizing...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset   = val_dataset.map(tokenize, batched=True)
    test_dataset  = test_dataset.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format("torch", columns=cols)
    val_dataset.set_format("torch", columns=cols)
    test_dataset.set_format("torch", columns=cols)

    return tokenizer, train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="weighted"),
        "f1_macro" : f1_score(labels, preds, average="macro"),
    }


def train(train_dataset, val_dataset, tokenizer):
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE * 2,
        warmup_steps                = 200,
        weight_decay                = 0.01,
        learning_rate               = LEARNING_RATE,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 50,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
        seed                        = 42,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n🚀 Starting training...")
    trainer.train()
    return trainer, model


def evaluate(trainer, test_dataset, test_df):
    print("\n📊 Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = test_df["label"].values

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    print(f"\n   Test Accuracy : {acc:.4f}")
    print(f"   Test F1       : {f1:.4f}")
    print("\n   Detailed Report:")
    print(classification_report(labels, preds,
          target_names=["Positive", "Negative", "Neutral"]))


def save_model(trainer, tokenizer):
    print(f"\n💾 Saving model to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    trainer.model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("   Saved locally ✓")

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_path = "/content/drive/MyDrive/indicbert-hinglish"
        trainer.model.save_pretrained(drive_path)
        tokenizer.save_pretrained(drive_path)
        print(f"   Saved to Google Drive ✓")
    except ImportError:
        print("   (Not in Colab — skipping Drive save)")


if __name__ == "__main__":
    print("=" * 55)
    print("   HINGLISH NLP — FINE-TUNING INDICBERT")
    print("=" * 55)

    train_df, val_df, test_df = load_data()
    tokenizer, train_ds, val_ds, test_ds = tokenize_data(train_df, val_df, test_df)
    trainer, model = train(train_ds, val_ds, tokenizer)
    evaluate(trainer, test_ds, test_df)
    save_model(trainer, tokenizer)

    print("\n🎉 Training complete!")
    print("   Download models/indicbert-hinglish/ from Colab")
    print("   Then run: streamlit run app.py")