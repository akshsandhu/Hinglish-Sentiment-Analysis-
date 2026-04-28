"""
conll_parser.py
───────────────
Parses SemEval 2020 Task 9 CoNLL format into a simple CSV.

The raw file looks like this (one word per line, blank line = new tweet):
    Meta  Meta  O  Positive
    kya   Hi    O
    scene En    O
    hai   Hi    O

This script:
  - Groups words into tweets (split on blank lines)
  - Grabs the sentence-level label from the first line's 4th column
  - Reconstructs the tweet text from the first column of each line
  - Saves a clean CSV with columns: text, label

Usage:
    python src/conll_parser.py
"""

import pandas as pd
import os


def parse_conll(filepath: str) -> pd.DataFrame:
    """
    Parse a SemEval CoNLL file into a DataFrame with (text, label) columns.

    Args:
        filepath: Path to the .txt CoNLL file

    Returns:
        pd.DataFrame with columns: text, label (0=Positive, 1=Negative, 2=Neutral)
    """
    label_map = {
        "positive": 0,
        "negative": 1,
        "neutral" : 2,
    }

    records   = []
    words     = []
    sentiment = None

    print(f"📂 Parsing: {filepath}")

    in_tweet = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Blank line = end of current tweet
            if line.strip() == "":
                if words and sentiment is not None:
                    text = " ".join(words)
                    records.append({"text": text, "label": sentiment})
                words     = []
                sentiment = None
                in_tweet  = False
                continue

            parts = line.split("\t")

            # Meta line: "meta  <id>  <label>"
            if parts[0].strip().lower() == "meta":
                if len(parts) >= 3:
                    label_str = parts[2].strip().lower()
                    sentiment = label_map.get(label_str, None)
                in_tweet = True
                continue

            # Word line: "<word>  <lang_tag>"
            if in_tweet and len(parts) >= 1:
                word = parts[0].strip()
                if word and word != "O":
                    words.append(word)

    # Catch last tweet if file doesn't end with blank line
    if words and sentiment is not None:
        records.append({"text": " ".join(words), "label": sentiment})

    df = pd.DataFrame(records)
    print(f"   ✅ Parsed {len(df)} tweets")

    if len(df) > 0:
        print(f"\n   Label distribution:")
        label_names = {0: "Positive", 1: "Negative", 2: "Neutral"}
        for label_id, count in df["label"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            print(f"   {label_names[label_id]:10s}: {count:5d} ({pct:.1f}%)")

    return df


if __name__ == "__main__":
    print("=" * 50)
    print("   HINGLISH NLP — CoNLL PARSER")
    print("=" * 50)

    input_path  = "data/train.txt"
    output_path = "data/labeled_reviews.csv"

    if not os.path.exists(input_path):
        print(f"\n❌ File not found: {input_path}")
        print("   Make sure you copied Hinglish_train_14k_split_conll")
        print("   into data/ and renamed it to train.txt")
        exit(1)

    df = parse_conll(input_path)

    if len(df) == 0:
        print("\n❌ No tweets parsed — check the file format")
        exit(1)

    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved → {output_path}")
    print(f"\n🎯 Next step: python src/preprocess.py")