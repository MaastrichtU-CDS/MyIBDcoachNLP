#!/usr/bin/env python3
"""
Embed texts using SentenceTransformers on Snellius GPU.

Automatically appends the model name to the output file to prevent overwriting.
"""

import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re


def sanitize_filename(s):
    """Remove characters that cannot be used in filenames."""
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings using SentenceTransformers.")
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx",
        help="Path to input Excel file containing text data."
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Column name in the Excel file containing the sentences."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings",   # now a *prefix*, not a full filename
        help="Output filename prefix (model name will be appended automatically)."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="specify Hugging Face model ID. E.g., 'NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers' or 'Qwen/Qwen3-Embedding-8B'."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Short name used for output directory and filenames (e.g. qwen3, robbert)."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join(args.output, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{args.model_name}_sentence_embeddings.npy"
    )

    print("=== Loading data ===")
    df = pd.read_excel(args.input)
    
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in input file.")
    
    sentences = (
        df[args.text_column]
        .astype(str)
        .fillna("")
        .tolist()
    )
    
    print(f"Loaded {len(sentences)} sentences.")

    print(f"=== Loading model: {args.model} ===")
    model = SentenceTransformer(args.model)

    print("=== Generating embeddings ===")
    embeddings = model.encode(
        sentences,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"=== Saving embeddings to {output_file} ===")
    np.save(output_file, embeddings)

    print("Done.")


if __name__ == "__main__":
    main()
