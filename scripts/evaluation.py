# evaluating cluster-mapped labels with annotations

import pandas as pd


def main():

    # load the annotation df and document info and topic info labeled
    annotation_df = pd.read_csv("data/annotations_for_assessment.csv")
    document_info = pd.read_csv("results/robbert/robbert_final_document_info.csv")
    topic_info_labeled = pd.read_csv("results/robbert/robbert_final_topic_info_labeled.csv")

    # first join document info and topic info labeled to get the message-level topics
    documents_with_labels = document_info.merge(topic_info_labeled[["Topic", "Translation", "Label"]], on="Topic", how="left")

    # then merge the annotation df with the documents_with_labels df
    merged_df = annotation_df.merge(documents_with_labels, on=["sentence_id"], how="left")

    # compare Label and Annotated_Label columns
    merged_df["Label_vs_Annotated_Label"] = merged_df.apply(lambda row: "Match" if row["Label"] == row["Annotated_Label"] else "Mismatch", axis=1)
    
    # print head of merged_df
    print(merged_df.head(10))

    # print the match rate
    match_rate = merged_df["Label_vs_Annotated_Label"].value_counts(normalize=True).get("Match", 0)
    print(f"Match rate: {match_rate:.2%}")
if __name__ == "__main__":
    main()