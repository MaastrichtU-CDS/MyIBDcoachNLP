from operator import index 
import pandas as pd 
from bertopic import BERTopic
# For the saved BERTopic model with reduced outliers, we do the following post-processing steps:
# 1. Merge the original duplicates back onto the topic assignment (document_info)
# 2. Recalculate the topic frequency with duplicates and merged topics
# 3. Save the final document_info and topic_info to CSV files for further analysis and visualization

# ---------- Restore the duplicated sentences to the pool of sentences ---------- 
def merge_with_full_sentences(doc_info, full_sentence_df): 
    """ Merge doc_info (unique sentences + topic assignment) with full sentence dataframe (with duplicates). """ 
    # If column is named "Document_Original", rename for merge 
    if "Document_Original" in doc_info.columns: 
        doc_info = doc_info.rename(columns={"Document_Original": "sentence"}) 
    if "Document" in doc_info.columns: 
        doc_info = doc_info.rename(columns={"Document": "sentence"})
    # Merge duplicates back onto topic assignment 
    merged = full_sentence_df.merge(doc_info, on='sentence', how='left' ) 
    return merged 

# ---------- UPDATED: Load model dfs and merge duplicates ---------- 
def merge_duplicates(full_sentence_df, df_doc_info, df_topic_info): 

    sorted_topic_info = df_topic_info.sort_values("Count", ascending=False) 

    # --- merge original duplicates back --- 
    merged_full_df = merge_with_full_sentences(df_doc_info, full_sentence_df) 
    # --- recalc topic frequency with duplicates --- 
    topic_counts = merged_full_df['Topic'].value_counts().sort_index() 
    # Replace Count column in topic_info 
    sorted_topic_info = sorted_topic_info.copy() 
    sorted_topic_info['Count'] = sorted_topic_info['Topic'].map(topic_counts).fillna(0).astype(int) 
    # sort the new Count column in descending order 
    sorted_topic_info = sorted_topic_info.sort_values("Count", ascending=False).reset_index(drop=True) 
    # add new column with frequency percentage 
    total_count = sorted_topic_info['Count'].sum() 
    sorted_topic_info["Frequency (%)"] = (sorted_topic_info['Count'] / total_count * 100).round(2)

    return sorted_topic_info, merged_full_df

def main(): 
    # input paths
    full_sentence_path = "./data/cleaned_patient_sentences.xlsx"
    document_sentence_path = "./data/cleaned_patient_deduplicated_sentences_for_embedding.xlsx" 
    reduced_model_path = "results/robbert/robbert_model_reduced"

    # output paths
    merged_full_df_path = "results/robbert/robbert_final_document_info.csv"
    sorted_topic_info_path = "results/robbert/robbert_final_topic_info.csv"

    # load the model with reduced outliers and its document info before merging
    reduced_model = BERTopic.load(reduced_model_path)

    # load the full sentence dataframe and reduced document info
    full_sentence_df = pd.read_excel(full_sentence_path) 
    sentence_df = pd.read_excel(document_sentence_path)
    sentences = sentence_df['sentence'].tolist()

    # access the document_info from the loaded model
    document_info = reduced_model.get_document_info(docs=sentences)
    topic_info = reduced_model.get_topic_info()


    # Merge duplicates back onto topic assignment and recalc topic frequency
    sorted_topic_info, merged_full_df = merge_duplicates(full_sentence_df, document_info, topic_info)
    # drop "Representative_Document" column from sorted_topic_info if it exists (not needed for final output)
    if "Representative_Document" in sorted_topic_info.columns:
        sorted_topic_info = sorted_topic_info.drop(columns=["Representative_Document"])

    # save
    merged_full_df.to_csv(merged_full_df_path, index=False)
    sorted_topic_info.to_csv(sorted_topic_info_path, index=False)
    print(f"Outputs saved.")   


if __name__ == "__main__": 
    main()
