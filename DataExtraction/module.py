import pandas as pd
import spacy
import torch
import re

def data_extractor(positive_reviews, negative_reviews, name):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use the first CUDA device
        print("Using CUDA device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")  # Use CPU if CUDA is not available
        print("CUDA is not available, using CPU")
    
    # Load spaCy model and activate GPU if available
    activated = spacy.require_gpu(0)
    print(activated)
    nlp = spacy.load("en_core_web_lg")

    # Function to extract nouns from text data
    def find_nouns(text_data, name):
        nouns = []  # List to store extracted nouns
        for text in text_data:
            # Remove the restaurant name from the text
            text = re.sub(name, "", text, flags=re.IGNORECASE)
            doc = nlp(text)  # Process text with spaCy
            for sentence in doc.sents:  # Iterate over sentences in the text
                for token in sentence:  # Iterate over tokens in the sentence
                    if token.pos_ == "NOUN":  # Check if the token is a noun
                        nouns.append(token.lemma_)  # Append the lemma of the noun to the list
        return nouns

    # Function to find the top ten key nouns
    def find_top_ten_keys(nouns):
        keywords_count = {}  # Dictionary to count occurrences of each noun
        for word in nouns:
            if word in keywords_count:
                keywords_count[word] += 1
            else:
                keywords_count[word] = 1
        # Sort the nouns by their count in descending order
        sorted_keywords = sorted(keywords_count.items(), key=lambda x: x[1], reverse=True)
        
        top_ten = [sorted_keywords[0]]  # Initialize top ten list with the most frequent noun
        i = 0
        # Iterate through sorted keywords to find top ten unique nouns
        while len(top_ten) < 10 and i < len(sorted_keywords) - 1:
            i += 1
            word1 = nlp(sorted_keywords[i][0])
            is_similar = False
            for word2 in top_ten:
                word2_doc = nlp(word2[0])
                similarity = word1.similarity(word2_doc)  # Compute similarity between nouns
                if similarity > 0.7:  # Check if nouns are similar
                    is_similar = True
                    break
            if not is_similar:
                top_ten.append(sorted_keywords[i])  # Add unique noun to top ten list
        return top_ten

    # Extract nouns from positive and negative reviews
    positive_label = find_nouns(positive_reviews['text'], name)
    negative_label = find_nouns(negative_reviews['text'], name)

    # Find top ten key nouns for positive and negative reviews
    positive_label = find_top_ten_keys(positive_label)
    negative_label = find_top_ten_keys(negative_label)

    # Return the top ten key nouns for positive and negative reviews
    return positive_label, negative_label
