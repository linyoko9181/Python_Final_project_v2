import pandas as pd
import spacy
import torch

def data_extractor(positive_reviews, negative_reviews):
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
    def find_nouns(text_data):
        nouns = []
        for text in text_data:
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
        keywords_count = sorted(keywords_count.items(), key=lambda x: x[1], reverse = True)
        top_ten = keywords_count[:10] 
        return top_ten


    positive_label = find_nouns(positive_reviews['text'])
    negative_label = find_nouns(negative_reviews['text'])

    positive_label = find_top_ten_keys(positive_label)
    negative_label  =find_top_ten_keys(negative_label)

    return positive_label, negative_label