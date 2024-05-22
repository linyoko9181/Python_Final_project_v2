import pandas as pd
import spacy
import torch
import re

def data_extractor(positive_reviews, negative_reviews, name):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Using CUDA device:", torch.cuda.get_device_name(device))
    else:
        print("CUDA is not available, using CPU")
    
    #load model
    activated = spacy.require_gpu(0)
    print(activated)
    nlp = spacy.load("en_core_web_lg")

    #extract noun from the sentence
    def find_nouns(text_data, name):
        nouns = []
        for text in text_data:
            text = re.sub(name, "", text, flags=re.IGNORECASE)
            doc = nlp(text)
            for sentence in doc.sents:
                for token in sentence:
                    if token.pos_ == "NOUN":
                        nouns.append(token.lemma_)
        return nouns

    #find the top ten key
    def find_top_ten_keys(nouns):
        keywords_count = {}
        for word in nouns:
            if word in keywords_count:
                keywords_count[word] += 1
            else:
                keywords_count[word] = 1
        sorted_keywords = sorted(keywords_count.items(), key=lambda x: x[1], reverse = True)
        
        top_ten = [sorted_keywords[0]]
        i = 0
        while len(top_ten) < 10 and i < len(sorted_keywords) - 1:
            i += 1
            word1 = nlp(sorted_keywords[i][0])
            is_similar = False
            for word2 in top_ten:
                word2_doc = nlp(word2[0])
                similarity = word1.similarity(word2_doc)
                if similarity > 0.7:
                    is_similar = True
                    break
            if not is_similar:
                top_ten.append(sorted_keywords[i])
        return top_ten


    positive_label = find_nouns(positive_reviews['text'], name)
    negative_label = find_nouns(negative_reviews['text'], name)

    positive_label = find_top_ten_keys(positive_label)
    negative_label  =find_top_ten_keys(negative_label)

    return positive_label, negative_label