from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-nl-en"

INPUT_CSV = Path("data/annotations_for_assessment.csv")
OUTPUT_CSV = Path("data/annotations_for_assessment_en.csv")

SOURCE_COLUMN = "sentence"
TARGET_COLUMN = "sentence_en"

BATCH_SIZE = 32
MAX_INPUT_TOKENS = 512


def choose_device() -> torch.device:
    """Select CUDA, Apple Silicon MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def translate_batch(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
) -> List[str]:
    """Translate a batch of Dutch strings into English."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def translate_column(series: pd.Series) -> pd.Series:
    """Translate non-empty values while preserving missing rows."""
    device = choose_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    values = series.fillna("").astype(str).tolist()
    translations = [""] * len(values)

    nonempty_indices = [
        index
        for index, text in enumerate(values)
        if text.strip()
    ]

    for start in tqdm(
        range(0, len(nonempty_indices), BATCH_SIZE),
        desc="Translating",
    ):
        batch_indices = nonempty_indices[start : start + BATCH_SIZE]
        batch_texts = [values[index] for index in batch_indices]

        batch_translations = translate_batch(
            batch_texts,
            tokenizer,
            model,
            device,
        )

        for index, translation in zip(
            batch_indices,
            batch_translations,
        ):
            translations[index] = translation

    return pd.Series(translations, index=series.index)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    if "csv" in INPUT_CSV.suffix.lower():
        dataframe = pd.read_csv(INPUT_CSV)
    elif "xls" in INPUT_CSV.suffix.lower():
        dataframe = pd.read_excel(INPUT_CSV)

    if SOURCE_COLUMN not in dataframe.columns:
        raise KeyError(
            f"Column {SOURCE_COLUMN!r} not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe[TARGET_COLUMN] = translate_column(
        dataframe[SOURCE_COLUMN]
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    print(f"Saved translated CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()