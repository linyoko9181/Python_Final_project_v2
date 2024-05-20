import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def sentiment_analyzer(reviews):

    ## CHECK IF CUDA IS AVAILABLE
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Using CUDA device:", torch.cuda.get_device_name(device))
    else:
        print("CUDA is not available, using CPU")

    def get_sentiment_score(review, device):
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to(device)
        tokens = tokenizer.encode(review, return_tensors='pt').to(device)
        result = model(tokens)
        return int(torch.argmax(result.logits))+1

    #get the sentiment score 
    reviews['sentiment'] = reviews['text'].apply(lambda x: get_sentiment_score(x[:512], device))
    sentiment_avg = reviews['sentiment'].mean()

    #create positive reviews dataframe and negative reviews dataframe and save it
    positive_reviews = reviews[reviews['sentiment']>3].copy()
    negative_reviews = reviews[reviews['sentiment']<3].copy()
    positive_reviews.to_csv('positive.csv', index=False)
    negative_reviews.to_csv('negative.csv', index=False)

    return positive_reviews, negative_reviews, sentiment_avg