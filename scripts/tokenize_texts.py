#!/usr/bin/env python3
import argparse
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize original messages using the same tokenizer settings as BERTopic CountVectorizer."
    )

    parser.add_argument(
        "--input-data",
        type=str,
        default="data/cleaned_patient_messages.xlsx",
        help="Excel or CSV file containing original messages."
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="message",
        help="Column containing the original unsplit patient messages."
    )

    parser.add_argument(
        "--input-stopwords",
        type=str,
        default="data/stopwords-nl-extended.txt",
        help="Text file with one stopword per line."
    )

    parser.add_argument(
        "--output-tokens",
        type=str,
        default="data/tokens/tokenized_original_messages.pkl",
        help="Output pickle file."
    )

    return parser.parse_args()


def remove_placeholders(text: str) -> str:
    return re.sub(r"\[[A-Z]+(?:_[A-Z]+)?\]", " ", text)


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Input file must be .xlsx, .xls, or .csv")


def main():
    args = parse_args()

    df = read_table(args.input_data)

    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in input data.")

    with open(args.input_stopwords, "r") as f:
        stopwords = [line.strip() for line in f if line.strip()]

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        min_df=2,
        ngram_range=(1, 1),
        token_pattern=r"\b[a-zA-Z]{3,}\b"
    )

    analyzer = vectorizer.build_analyzer()

    messages = df[args.text_column].fillna("").astype(str).tolist()

    tokenized_messages = [
        analyzer(remove_placeholders(message))
        for message in messages
    ]

    # Remove empty tokenized messages
    tokenized_messages = [
        tokens for tokens in tokenized_messages
        if len(tokens) > 0
    ]

    with open(args.output_tokens, "wb") as f:
        pickle.dump(tokenized_messages, f)

    print(f"Saved {len(tokenized_messages)} tokenized messages to {args.output_tokens}")


if __name__ == "__main__":
    main()