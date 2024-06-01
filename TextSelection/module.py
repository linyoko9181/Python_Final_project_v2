import spacy
import random
import pandas as pd

def text_selector(sentiment, string):
    # Load spaCy's English model
    nlp = spacy.load("en_core_web_lg")

    # Function to find sentences containing the specified word
    def find_labels(text_data, string):
        sentence_index = []  # List to store indices of sentences containing the specified word
        for i, text in enumerate(text_data):
            doc = nlp(text)  # Process text with spaCy
            for sentence in doc.sents:  # Iterate over sentences in the text
                for token in sentence:  # Iterate over tokens in the sentence
                    if token.lemma_ == string:  # Check if the lemma of the token matches the specified word
                        sentence_index.append(i)
                        break  # No need to continue searching this sentence
        return sentence_index

    print(string)
    
    # Determine the file name based on sentiment (1 for positive, otherwise negative)
    if sentiment == 1:
        file_name = "positive.csv"
    else:
        file_name = "negative.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)

    # Access the column containing text data
    text_data = df['text']

    # Find sentences containing the specified word
    sentence_index = find_labels(text_data, string)
    
    # Choose a random sentence index from the list
    random_index = random.choice(sentence_index)

    # Retrieve the text corresponding to the random index
    random_sentence = text_data.iloc[random_index]

    print(random_sentence)
    return random_sentence
