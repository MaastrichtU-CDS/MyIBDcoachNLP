import pandas as pd
import os
import re

def aggregate_topics(document_info, message_df, sentence_df):
    # keep only relevant columns
    reduced_sentence_df = sentence_df[["message_id", "sentence_id", "sentence", "translated_sentence"]]

    # create sentence_id starting from 1 to align with sentence_df
    document_info = (
        document_info
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "sentence_id"})
    )
    document_info["sentence_id"] = document_info["sentence_id"] + 1

    # sanity check
    assert reduced_sentence_df["sentence_id"].min() == 1
    assert len(reduced_sentence_df) == len(document_info)

    # now merge by sentence_id
    merged_df = reduced_sentence_df.merge(document_info, on="sentence_id", how="inner")

    # filter messages to only "From client"
    filtered_message_df = message_df.loc[
        message_df["direction"] == "From client",
        ["clean_message", "message_id", "translated_message"]
    ]

    # group topics by message_id (as sets)
    grouped = merged_df.groupby("message_id")["Name"].agg(lambda x: set(x.dropna()))

    # --- pre-create all 120 topic columns ---
    all_topics = sorted(document_info["Name"].dropna().unique().tolist())
    if len(all_topics) != 120:
        print(f"⚠️ Warning: expected 120 topics, found {len(all_topics)} in document_info")

    # one-hot encode topics
    topics_ohe = grouped.apply(lambda topics: pd.Series(1, index=topics))
    topics_ohe = topics_ohe.fillna(0).astype(int)

    # reindex to make sure all 120 topic columns exist, in order
    topics_ohe = topics_ohe.reindex(columns=all_topics, fill_value=0)

    # count topics per message
    topics_ohe["number_of_topics_per_message"] = topics_ohe.sum(axis=1)

    # add column with list of topics
    topics_ohe["topics_list"] = grouped.apply(lambda s: sorted(list(s)))

    # join with filtered_message_df
    aggregated_df = filtered_message_df.merge(
        topics_ohe,
        left_on="message_id",
        right_index=True,
        how="left"
    )

    # --- reorder columns ---
    # current: [clean_message, message_id, translated_message, <topics...>]
    cols = list(aggregated_df.columns)

    # find indices
    fixed_cols = ["clean_message", "message_id", "translated_message", "topics_list", "number_of_topics_per_message"]
    topic_cols = [c for c in cols if c not in fixed_cols]

    # reorder: first 3 → add topics_list → add number_of_topics_per_message → then one-hot topics
    new_order = ["clean_message", "message_id", "translated_message", "topics_list", "number_of_topics_per_message"] + topic_cols
    aggregated_df = aggregated_df[new_order]

    return merged_df, aggregated_df


def main():
    # import the sentence df
    sentence_df = pd.read_excel("./data/sentence_data_for_analysis.xlsx", index_col=0)
    # import the message df
    message_df = pd.read_excel("./data/translated_clean_message_data.xlsx", index_col=0)
    # import the document df
    document_info = pd.read_csv("./new_analysis/results/models/robbert_final/doc_info_final.csv", index_col=0)
    # define the output dir
    output_dir = "new_analysis/results/tables"

    # check alignment
    if len(sentence_df) == len(document_info):
        merged_df, aggregated_df = aggregate_topics(document_info, message_df, sentence_df)
        # save merged df
        merged_df.to_csv(os.path.join(output_dir, "sentence_with_topic_info_robbert.csv"), index=False)
        # save the aggregated df
        aggregated_df.to_csv(os.path.join(output_dir, "aggregated_topics_robbert.csv"), index=False)
    else:
        print("Documents and sentences do not match.")


if __name__ == "__main__":
    main()
