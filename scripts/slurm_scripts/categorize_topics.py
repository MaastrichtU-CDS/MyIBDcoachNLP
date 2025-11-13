import pandas as pd

def clean_topic_name(topic_str):
    """
    Convert '0_bedankt_dank_reactie_alvast' →
    'Topic 0: bedankt, dank, reactie, alvast'
    """
    parts = topic_str.split("_")
    topic_num = parts[0]
    words = ", ".join(parts[1:])
    return f"Topic {topic_num}: {words}"

def map_category_and_translations(mapping_file, translations_file, topic_info_file, output_file):
    """
    Merge topic categories and English translations into topic_info.
    
    Args:
        mapping_file (str): path to topic_category_map.txt (topic_name,category)
        translations_file (str): path to topic_translations.txt (Topic x: English words)
        topic_info_file (str): path to topic_info_final.csv
        output_file (str): path to save merged CSV
    """
    # 1. Load mapping
    df_map = pd.read_csv(mapping_file)

    # add grouped categories
    df_map["category_grouped"] = df_map["category"].apply(
        lambda x: "medical" if x == "medical" else "non-medical"
    )

    # 2. Load translations (txt with one line per topic)
    translations = {}
    with open(translations_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Topic"):
                topic_num = line.split(":")[0].split()[1]  # e.g. Topic 5 -> "5"
                translations[int(topic_num)] = line  # store full English translation string

    df_trans = pd.DataFrame(
        [{"topic_num": k, "translation": v} for k, v in translations.items()]
    )

    # 3. Load topic_info
    topic_df = pd.read_csv(topic_info_file, index_col=0)

    # extract topic number from "Name" (assuming Name like '5_afspraak_poli...')
    topic_df["topic_num"] = topic_df["Name"].str.split("_").str[0].astype(int)

    # 4. Merge all
    merged = topic_df.merge(
        df_map[["topic_name", "category_grouped"]],
        how="left",
        left_on="Name",
        right_on="topic_name"
    ).merge(
        df_trans,
        how="left",
        on="topic_num"
    )

    # 5. Clean up
    merged = merged.drop(columns=["topic_name"])
    merged.to_excel(output_file, index=False)

    print(f"Saved enriched topic_info with categories + translations to {output_file}")
    return merged

# Example usage:
if __name__ == "__main__":
    merged = map_category_and_translations(
        mapping_file="topic_category_map.txt",
        translations_file="topic_translations.txt",
        topic_info_file="./results/models/robbert_final/topic_info_final.csv",
        output_file="./results/models/robbert_final/topic_info_with_categories_and_translations.xlsx"
    )
