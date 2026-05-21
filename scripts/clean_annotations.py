# read "annotations.csv"
import pandas as pd

anno_df = pd.read_csv("data/annotations_test.csv")
# replace "_x000d_" with "" in the whole dataframe
anno_df = anno_df.replace(r"_x000d_", "", regex=True)
# for "Manual_label_Tom", if it starts with "A" then set it to "A", if it starts with "M" then set it to "M"
label_col = "Manual_label_Tom"
anno_df[label_col] = anno_df[label_col].apply(lambda x: "A" if str(x).startswith("A") else ("M" if str(x).startswith("M") else x))

import pandas as pd
from rapidfuzz import process, fuzz

# Adjust names if needed
x_col = "sentence"
y_col = "Sentence"
label_col = "Manual_label_Tom"
# Clean text for better matching
def clean_text(s):
    if pd.isna(s):
        return ""
    return str(s).lower().strip()

anno_df["x_clean"] = anno_df[x_col].map(clean_text)
anno_df["y_clean"] = anno_df[y_col].map(clean_text)

# Sentence_Y -> label lookup
y_df = (
    anno_df[[y_col, "y_clean", label_col]]
    .dropna(subset=[y_col])
    .drop_duplicates(subset=["y_clean"], keep="first")
)

choices = y_df["y_clean"].tolist()
label_lookup = dict(zip(y_df["y_clean"], y_df[label_col]))

# Fuzzy match each Sentence_X against Sentence_Y
def fuzzy_map_label(x, threshold=90):
    if not x:
        return pd.Series([None, None, None])

    match = process.extractOne(
        x,
        choices,
        scorer=fuzz.token_sort_ratio
    )

    if match is None:
        return pd.Series([None, None, None])

    matched_sentence, score, _ = match

    if score >= threshold:
        return pd.Series([
            matched_sentence,
            score,
            label_lookup.get(matched_sentence)
        ])

    return pd.Series([matched_sentence, score, None])

anno_df[["Matched_Sentence_Y_clean", "Match_Score", "Mapped_Label"]] = (
    anno_df["x_clean"].apply(fuzzy_map_label)
)

# drop x_clean and y_clean
anno_df = anno_df.drop(columns=["x_clean", "y_clean"])
# drop where Mapped_Label is null
anno_df = anno_df.dropna(subset=["Mapped_Label"])
anno_df.to_csv("data/annotations_with_fuzzy_labels.csv", index=False)