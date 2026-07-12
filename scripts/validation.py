# evaluating cluster-mapped labels with annotations

import pandas as pd


def main():
    # load the annotation df and document info and topic info labeled
    annotation_df = pd.read_csv("data/annotations_for_assessment_en.csv") # dataframe containing the annotated labels for the sentences
    document_info = pd.read_csv("results/robbert/robbert_final_document_info.csv") # data frame containing the document-level topic information
    topic_info_labeled = pd.read_csv("results/robbert/robbert_final_topic_info_labeled.csv") # data frame containing the topic-level information with labels

    # change "Medical" to "M" AND "Non-Medical" to "A" in the Label column of topic_info_labeled
    topic_info_labeled["Label"] = topic_info_labeled["Label"].replace({"Medical": "M", "Non-Medical": "A"})
    # first join document info and topic info labeled to get the message-level topics
    documents_with_labels = document_info.merge(topic_info_labeled[["Topic", "Translation", "Label"]], on="Topic", how="left")

    # then merge the annotation df with the documents_with_labels df
    merged_df = annotation_df.merge(documents_with_labels, on=["sentence_id"], how="left")
    # drop NA in merged df for "Label" and "Annotated_Label" columns
    merged_df = merged_df.dropna(subset=["Label", "Annotated_Label"])
    # compare Label and Annotated_Label columns
    merged_df["Label_vs_Annotated_Label"] = merged_df.apply(lambda row: "Match" if row["Label"] == row["Annotated_Label"] else "Mismatch", axis=1)

    # print the fraction of matches and mismatches
    match_count = merged_df["Label_vs_Annotated_Label"].value_counts()
    match_fraction = match_count / len(merged_df)
    print("Match fraction on sentence level:")
    print(match_fraction)

if __name__ == "__main__":
    main()