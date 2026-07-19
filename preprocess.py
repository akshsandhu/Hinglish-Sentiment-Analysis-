"""
preprocess.py
─────────────
Hinglish text cleaning and dataset preparation.
Run AFTER conll_parser.py.

Usage:
    python src/preprocess.py
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
import os

SLANG_MAP = {
    "nhi"       : "nahi",
    "nahi"      : "nahi",
    "ni"        : "nahi",
    "bhut"      : "bahut",
    "boht"      : "bahut",
    "bht"       : "bahut",
    "accha"     : "accha",
    "acha"      : "accha",
    "achha"     : "accha",
    "thik"      : "theek",
    "theek"     : "theek",
    "tk"        : "theek",
    "kafi"      : "kaafi",
    "kaafi"     : "kaafi",
    "zyada"     : "zyada",
    "jyada"     : "zyada",
    "mast"      : "mast",
    "zabardast" : "zabardast",
    "shandar"   : "shandar",
    "bekar"     : "bekar",
    "bakwas"    : "bakwas",
    "bakwaas"   : "bakwas",
    "badhiya"   : "badhiya",
    "badia"     : "badhiya",
    "badhia"    : "badhiya",
    "sahi"      : "sahi",
    "galat"     : "galat",
    "kharab"    : "kharab",
    "aur"       : "aur",
    "or"        : "aur",
    "hai"       : "hai",
    "h"         : "hai",
    "yeh"       : "yeh",
    "ye"        : "yeh",
    "woh"       : "woh",
    "vo"        : "woh",
    "bilkul"    : "bilkul",
    "ekdum"     : "ekdum",
}

LABEL_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "pos": 0,
    "neg": 1,
    "neu": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}
VALID_LABEL_IDS = frozenset(LABEL_MAP.values())


def normalize_sentiment_label(value):
    """Normalize text, integer, and integer-valued float labels."""
    normalized = str(value).casefold().strip()
    mapped = LABEL_MAP.get(normalized)
    if mapped is not None:
        return mapped

    try:
        numeric = float(normalized)
    except ValueError:
        return None

    if numeric.is_integer() and int(numeric) in VALID_LABEL_IDS:
        return int(numeric)
    return None


def clean_hinglish_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10}\b", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\u0900-\u097F\w\s!?,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    words = text.split()
    words = [SLANG_MAP.get(word, word) for word in words]
    return " ".join(words)


def prepare_dataset(df: pd.DataFrame, data_dir: str = "data") -> tuple:
    print("\n🧹 Cleaning text...")

    df.columns = [c.lower().strip() for c in df.columns]

    # Handle alternate column names
    for col in ["tweet", "sentence", "review", "content"]:
        if col in df.columns and "text" not in df.columns:
            df = df.rename(columns={col: "text"})

    df["clean_text"] = df["text"].apply(clean_hinglish_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
    print(f"   Removed {before - len(df)} empty rows after cleaning")

    df["label"] = df["label"].apply(normalize_sentiment_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("\n📊 Label distribution:")
    label_names = {0: "Positive", 1: "Negative", 2: "Neutral"}
    for label_id, count in df["label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"   {label_names[label_id]:10s}: {count:5d} ({pct:.1f}%)")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, random_state=42,
        stratify=train_df["label"]
    )

    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(f"{data_dir}/train.csv", index=False)
    val_df.to_csv(f"{data_dir}/val.csv",     index=False)
    test_df.to_csv(f"{data_dir}/test.csv",   index=False)

    print("\n✅ Splits saved:")
    print(f"   Train : {len(train_df)} rows → data/train.csv")
    print(f"   Val   : {len(val_df)} rows → data/val.csv")
    print(f"   Test  : {len(test_df)} rows → data/test.csv")

    return train_df, val_df, test_df


if __name__ == "__main__":
    print("=" * 50)
    print("   HINGLISH NLP - PREPROCESSING")
    print("=" * 50)

    input_file = "data/labeled_reviews.csv"

    if not os.path.exists(input_file):
        print(f"\n❌ {input_file} not found.")
        print("   Run this first:  python src/conll_parser.py")
        exit(1)

    df = pd.read_csv(input_file)
    print(f"\n📂 Loaded {len(df)} rows from {input_file}")

    prepare_dataset(df, "data")

    print("\n🎉 Preprocessing complete!")
    print("   Next step: Open Google Colab and run src/train_colab.py")
