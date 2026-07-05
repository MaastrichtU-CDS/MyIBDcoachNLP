#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run all BERTopic grid-search chunks locally.")

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--chunk-dir", default="chunks")
    parser.add_argument("--embeddings-dir", default="embeddings")
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--input-data", default="data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx")
    parser.add_argument("--input-tokens", default="data/tokens/tokenized_sentences.pkl")
    parser.add_argument("--input-stopwords", default="data/stopwords-nl-extended.txt")

    return parser.parse_args()


def main():
    args = parse_args()

    embedding_files = list(Path(args.embeddings_dir).glob(f"*{args.model_name}*.npy"))
    if not embedding_files:
        raise FileNotFoundError(
            f"No embedding file found in {args.embeddings_dir} matching *{args.model_name}*.npy"
        )

    embedding_file = embedding_files[0]
    print(f"Using embedding file: {embedding_file}")

    chunk_files = sorted(Path(args.chunk_dir).glob("chunk_*.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.json files found in {args.chunk_dir}")

    print(f"Found {len(chunk_files)} chunks.")

    for chunk_file in chunk_files:
        chunk_id = int(chunk_file.stem.split("_")[1])

        print(f"\nRunning chunk {chunk_id}")

        cmd = [
            "python", "scripts/run_grid_chunk.py",
            "--model-name", args.model_name,
            "--chunk-id", str(chunk_id),
            "--run-id", args.run_id,
            "--input-embeddings", str(embedding_file),
            "--chunk-dir", args.chunk_dir,
            "--output-dir", args.output_dir,
            "--input-data", args.input_data,
            "--input-tokens", args.input_tokens,
            "--input-stopwords", args.input_stopwords,
        ]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()