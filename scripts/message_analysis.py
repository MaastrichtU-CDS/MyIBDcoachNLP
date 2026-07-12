# analyzing topics on a message level
import os
import itertools
import pandas as pd

def aggregate_topics_by_message(document_info, topic_info_labeled):
    """
    Aggregate topics by message_id and return a new DataFrame with message_id and the list of unique topics.
    """

    # merge the document_info and topic_info DataFrames on "Topic"
    merged_df = document_info.merge(topic_info_labeled[["Topic", "Translation", "Label"]], on="Topic", how="left")
    
    # return merged_df
    return merged_df

def topic_coherence_analysis(merged_df, excluded_topics, output_dir="results/robbert"):
    
    os.makedirs(output_dir, exist_ok=True)

    # Remove rows without topic labels if any exist
    merged_df = merged_df.dropna(subset=["Topic"]).copy()
    # Remove excluded topics
    merged_df = merged_df[
        ~merged_df["Topic"].isin(excluded_topics)
    ].copy()

    # Make sure Topic is integer-like
    merged_df["Topic"] = merged_df["Topic"].astype(int)

    # Create readable topic name
    merged_df["Topic_Name"] = (
        merged_df["Topic"].astype(str) + ": " + merged_df["Translation"].astype(str)
    )
    message_topics = (
        merged_df.groupby("message_id", sort=False)
        .agg(
            unique_topics=("Topic", lambda x: sorted(set(x))),
            unique_topic_names=("Topic_Name", lambda x: sorted(set(x))),
            unique_labels=("Label", lambda x: sorted(set(x.dropna()))),
            n_unique_topics=("Topic", "nunique"),
            n_sentences=("Topic", "size"),
        )
        .reset_index()
    )

    message_topics["topic_combination"] = message_topics["unique_topics"].apply(
        lambda x: " | ".join(map(str, x))
    )

    message_topics["topic_name_combination"] = message_topics["unique_topic_names"].apply(
        lambda x: " | ".join(x)
    )

    # print the occurences of "[Non-Medical]" in the unique_labels column
    message_topics.to_csv(
        os.path.join(output_dir, "message_level_topics.csv"),
        index=False,
    )


    # Topic pair co-occurrence
    topic_pairs = []

    for _, row in message_topics.iterrows():
        topics = row["unique_topic_names"]

        if len(topics) > 1:
            for topic_a, topic_b in itertools.combinations(topics, 2):
                topic_pairs.append((topic_a, topic_b))

    topic_pair_df = pd.DataFrame(topic_pairs, columns=["topic_a", "topic_b"])

    if not topic_pair_df.empty:
        topic_cooccurrence = (
            topic_pair_df.value_counts()
            .reset_index(name="n_messages")
            .sort_values("n_messages", ascending=False)
        )
    else:
        topic_cooccurrence = pd.DataFrame(
            columns=["topic_a", "topic_b", "n_messages"]
        )

    topic_cooccurrence.to_csv(
        os.path.join(output_dir, "topic_cooccurrence_pairs.csv"),
        index=False,
    )
    print("\n=== Message-level topic analysis ===")
    print(f"Number of messages: {len(message_topics)}")
    print(f"Median unique topics per message: {message_topics['n_unique_topics'].median()}")
    print(f"Mean unique topics per message: {message_topics['n_unique_topics'].mean():.2f}")
    print(f"Max unique topics per message: {message_topics['n_unique_topics'].max()}")

    print("\nTop 10 topic co-occurrences:")
    print(topic_cooccurrence.head(10))

    print("\nUnique labels on the message level:")
    print(message_topics["unique_labels"].value_counts())
    print(f"\nSaved outputs to: {output_dir}")

    return message_topics, topic_cooccurrence

def main():
    # input files
    document_info = pd.read_csv(f"results/robbert/robbert_final_document_info.csv")
    topic_info_labeled = pd.read_csv(f"results/robbert/robbert_final_topic_info_labeled.csv")
    excluded_topics = pd.read_csv("results/robbert/robbert_topics_excluded_from_visualization.csv")["Topic"].tolist()

    # merge the document_info and topic_info DataFrames on "Topic"
    merged_df = aggregate_topics_by_message(document_info, topic_info_labeled)

    # analyzie topic coherence on a message level
    topic_coherence_analysis(merged_df, excluded_topics)

    
if __name__ == "__main__":
    main()
