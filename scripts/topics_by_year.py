# analyzing the most common topics per year in the dataset
import pandas as pd

def main():
    # Load the dataset
    topic_info_labeled = pd.read_csv("results/robbert/robbert_final_topic_info_labeled.csv")
    document_info = pd.read_csv("results/robbert/robbert_final_document_info.csv")
    messages_with_dates = pd.read_excel("data/cleaned_patient_messages.xlsx")


if __name__ == "__main__":
    main()