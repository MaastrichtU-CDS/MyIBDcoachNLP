from bs4 import BeautifulSoup
import re
import argparse
import pandas as pd
from nltk.tokenize import sent_tokenize

def remove_html_tags(text: str) -> str:
    ''' Arguments:
    text: string from which html tags need to be removed
    Returns: string without html tags
    '''
    return BeautifulSoup(text, "html.parser").get_text()

def replace_numbered_patterns(text):
    text = re.sub(r'\[(\w+)-\d+\]', r'[\1]', text, flags=re.IGNORECASE) # replace numbered patterns 
    text = re.sub(r'\[person\]', '[PERSOON]', text, flags=re.IGNORECASE) # standardize the English [PERSON] with Dutch [PERSOON]
    text = re.sub(r'\[INSTITUTION\]', '[ZORGINSTELLING]', text, flags=re.IGNORECASE) # replace institutioin to zorginstelling
    text = re.sub(r':\)|:\(|;\)', '', text) # removes emojis
    text = re.sub(r"^[^\w([]+", '', text) # removes leading punctuations
    # Match any pattern of the form [WORD-<number>] and replace with [WORD]
    return text

# Optional: function to remove entities in placeholders, not used currently
'''
def remove_entities(text):
    # entity labels to remove (case-insensitive)
    entity_labels = [
        "persoon", "locatie", "url", "telefoonnummer",
        "emailadres", "datum", "zorginstelling"
    ]
    # remove known entity labels
    entity_pattern = r'\[(' + "|".join(entity_labels) + r')\]'

    text = re.sub(entity_pattern, '', text, flags=re.IGNORECASE)

    # remove simple emoticons
    text = re.sub(r':\)|:\(|;\)', '', text)

    # remove leading punctuation except '['
    text = re.sub(r"^[^\w([]+", '', text)

    return text.strip()
'''

def remove_greetings_and_signoffs(text: str) -> str:
    greetings_signoffs_patterns = [
        r"^\s*(beste|geachte|dear|hallo|hoi|hoi hoi|goedemorgen|goede morgen|goedemiddag|goede?n middag|goede?n avond|goedenavond|dag|hello|hi|morning|good morning|good afternoon|good evening)\s*[,!?.]*\s*"
        r"\b(best regards|regards|groe?t|groe?ten|gr|groetjes?|grtjs?|mvg|m?e?t?\s*vriendelijke?\s*gr\w*|met\s*vriendelijke?|hartelijke\s*gr\w*)\b\s*[,!?.]*\s*"
  # sign-offs at end
    ]
    for pattern in greetings_signoffs_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def clean_messages(messages: list) -> list[str]:
    ''' Arguments:
    sentences: list of messages to be cleaned, already anonymized but needs further cleaning
    Returns: list of cleaned sentences
    '''

    # Step 1: Remove html tags
    print("Step 1: Removing HTML tags...")
    cleaned_messages = [remove_html_tags(message) for message in messages]

    # Step 2: Replace numbered patterns in entity placeholders (There is an optional step to remove entities, currently not used)
    print("Step 2: Replacing numbered patterns in entity placeholders...")
    cleaned_messages = [replace_numbered_patterns(message) for message in cleaned_messages]

    # Step 3: Remove greetings and signoffs
    print("Step 3: Removing greetings and sign-offs...")
    cleaned_messages = [remove_greetings_and_signoffs(message) for message in cleaned_messages]

    return cleaned_messages

def clean_message_df(df):
    ''' Arguments:
    df: dataframe containing messages FROM PATIENTS to be cleaned, this dataframe must contain columns "user_guid", "date", "message", "direction"
    message_column: name of the column in the dataframe that contains the messages
    Returns: list of cleaned messages
    '''
    print("\n=============Cleaning patient messages=================")


    # print the number of messages in the dataframe
    print(f"Number of messages from patients before cleaning: {len(df[df['direction'] == 'From client'])}")
    # print the number of unique users in the dataframe
    print(f"Number of unique patient users in dataframe: {df[df['direction'] == 'From client']['user_guid'].nunique()}")
    # print the time range of the messages in the dataframe
    print(f"Time range of messages: {df['date'].min()} to {df['date'].max()}")

    # Step 2: clean the messages from patients and add a new column "message_cleaned"
    cleaned_messages = clean_messages(df[df['direction'] == 'From client']["message"].tolist())
    df["message_cleaned"] = cleaned_messages

    # Step 3: drop rows where "message_cleaned" is empty
    print("Step 4: Dropping empty messages after cleaning...")
    df = df[df["message_cleaned"].str.strip() != ""]
     # create a new column "pat_message_id" as a unique identifier for each message from patients
    print(f"Number of messages from patients after cleaning: {len(df)}")

    return df

def get_patient_messages(df):
    ''' Arguments:
    df: dataframe containing messages, must contain column "direction"
    Returns: dataframe containing only messages from patients
    '''
    # renaming the columns to standard names
    if len(df.columns) != 7:
        raise ValueError("Dataframe must contain exactly 7 columns")
    df=df[2:]
    df.columns = ["provider_name", "program_name", "user_guid", "date", "message", "direction", "sender"]

    # create a message id column to keep track of messages
    df = df.copy()
    df.loc[:, "message_id"] = ["msg_" + str(i + 1) for i in range(len(df))]

    # check if the df has columns "user_guid", "date", "message" and "direction"
    if not all(col in df.columns for col in ["user_guid", "date", "message", "direction"]):
        raise ValueError("Dataframe must contain columns: user_guid, date, message, direction")
    patient_df = df[df['direction'] == 'From client']
    print("Number of total messages in dataset:", len(df))
    print("Time range of messages in dataset:", df['date'].min(), "to", df['date'].max())


    return patient_df

def create_sentence_data(messages_df):
    ''' Arguments:
    messages_df: dataframe containing
     cleaned messages from patients, must contain column "message_cleaned" and "message_id"
    Returns: dataframe containing sentences split from messages
    '''

    # two helper functions to split long sentences
    def split_sentence_with_matches(sentence, matches):
        parts = []
        for i in range(len(matches)):
            match = matches[i]
            part = sentence.split(match, 1)[0]
            
            if i > 0:
                part = "".join([matches[i-1], part])
            
            parts.append(part)
            remaining_part = sentence.split(match, 1)[1]
            sentence = remaining_part
        
        if remaining_part:
            part = "".join([matches[-1], remaining_part])
            parts.append(part)

        return parts

    def split_sentence(sentence, max_length=400):
        pattern1 = r'(?<!^)Maar|En|Ik|Ze|Het'
        pattern2 = r'(?<!^)\.\s*[A-Z][a-z]+'
        matches1 = re.findall(pattern1, sentence)
        matches2 = re.findall(pattern2, sentence)
        matches = matches1
        if len(matches1) < len(matches2):
            matches = matches2
        smaller_sentences = []
        if matches:
            smaller_sentences = split_sentence_with_matches(sentence, matches)
        else:
            #print("Can't split the following sentence: ", "\n", sentence)
            pass
        return smaller_sentences
    
    # List to hold the split sentences along with their message IDs
    sentence_data = []
    
    # Loop through each message in the original DataFrame
    print("\n==========Splitting messages into sentences with NLTK=========")
    print("Step 1: Automatic splitting with NLTK sentence tokenizer...")
    print("Step 2: Manual splitting of long sentences exceeding 400 characters...")
    print("Step 3: Correcting incorrect splits due to abbreviations...")
    for message_id, message in zip(messages_df['message_id'], messages_df["message_cleaned"]):
        # first replace any "incl." with "inclusief" to avoid incorrect splits
        message = re.sub(r'incl\.', 'inclusief', message)
        sentences = sent_tokenize(message)
            
        # Append each sentence along with its existing Message_ID
        for sentence in sentences:
            
            sentence = sentence.strip()
            if len(sentence) <= 400 or 'voedinginformatie' in sentence.lower():
                sentence_data.append({'message_id': message_id, 'sentence': sentence})

            else:
                #print("\nOriginal: ", "\n", sentence, "\n")
                # Call split_sentence to get smaller sentences
                smaller_sentences = split_sentence(sentence)
                
                if smaller_sentences:                    
                    for smaller_sentence in smaller_sentences:
                        smaller_sentence = re.sub(r"^[,!.?\]]*\s*", "", smaller_sentence)

                        sentence_data.append({'message_id': message_id, 'sentence': smaller_sentence})

                else:
                    pass
            

    # Create a new DataFrame with the split sentences
    sentence_df = pd.DataFrame(sentence_data)
    sentence_df = sentence_df.dropna(subset=['sentence']) # drop rows with NaN sentences

    # some splits are incorrect, we do a final cleaning step here by merging incorrect splits back together
    # find sentences ending in the pattern "a.b.c. "
    abbreviation_pattern = r"\b[a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]\.\s*$"
    to_drop = []
    for i, row in sentence_df[:-1].iterrows():
        sentence = row["sentence"]
        if re.search(abbreviation_pattern, sentence, flags=re.IGNORECASE):
            sentence_df.loc[i, "sentence"] += " " + sentence_df.loc[i + 1, "sentence"]

            to_drop.append(i + 1)
    # Drop merged sentences from DataFrame
    sentence_df.drop(to_drop, inplace=True)

    # Reset index
    sentence_df.reset_index(drop=True, inplace=True)
    print("Total number of sentences after splitting:", len(sentence_df))
    
    # now cleean the sentences again to remove any leading/trailing spaces or punctuations
    sentence_df['sentence'] = sentence_df['sentence'].apply(lambda x: re.sub(r"^[,!.?\]]*\s*", "", x).strip())

    # drop sentences of <= 15 characters
    print("Step 4: Final cleaning and dropping sentences with length <= 15 characters...")
    sentence_df = sentence_df[sentence_df['sentence'].str.len() > 15]
    # add a new column sentence_id as unique identifier for each sentence
    sentence_df["sentence_id"] = ["sent_" + str(i + 1) for i in range(len(sentence_df))]
    print("⭐ Number of sentences after final cleaning:", len(sentence_df))

    return sentence_df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset containing messages", default="data/anonymized_berichten.xlsx")
    parser.add_argument("--output_path", type=str, required=False,
                        help="Path to save the cleaned dataset", default="data/cleaned_patient_messages.xlsx")
    return parser.parse_args()

def main():
    # read the messages (they have already been anonymized)
    args = parse_args()
    # Read the dataset *here*
    data = pd.read_excel(args.data)
    print("Dataset loaded.")

    import nltk
    #nltk.download('punkt_tab') # download the punkt tokenizer if not already downloaded
    
    # 1. Filter the data to get only the messages from patients
    patient_data = get_patient_messages(data)

    # 2. Clean the messages
    cleaned_messages_df = clean_message_df(patient_data)
    # Save the cleaned messages to a new file
    cleaned_messages_df.to_excel(args.output_path, index=False)
    print("✅ Messages cleaned and saved to", args.output_path)

    # 3. Split the cleaned messages into sentences and prepare for analysis
    sentence_df = create_sentence_data(cleaned_messages_df)
    # save the sentence dataframe to a new file
    sentence_output_path = args.output_path.replace(".xlsx", "_sentences_for_analysis.xlsx")
    sentence_df.to_excel(sentence_output_path, index=False)
    print("✅ Sentences extracted and saved to", sentence_output_path)

    # 4. Translate the sentences using an external translation script (not included here)


if __name__ == "__main__":
    main()

