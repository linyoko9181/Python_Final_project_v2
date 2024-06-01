import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def sentiment_analyzer(reviews):
    ## CHECK IF CUDA IS AVAILABLE
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Use the first CUDA device
        print("Using CUDA device:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")  # Use CPU if CUDA is not available
        print("CUDA is not available, using CPU")

    # Function to get the sentiment score for a single review
    def get_sentiment_score(review, device):
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment').to(device)
        tokens = tokenizer.encode(review, return_tensors='pt').to(device)
        result = model(tokens)
        return int(torch.argmax(result.logits)) + 1  # Return sentiment score (1 to 5)

    # Apply sentiment analysis to each review (truncate to 512 characters if necessary)
    reviews['sentiment'] = reviews['text'].apply(lambda x: get_sentiment_score(x[:512], device))
    sentiment_avg = reviews['sentiment'].mean()  # Calculate average sentiment score

    # Create positive and negative reviews dataframes
    positive_reviews = reviews[reviews['sentiment'] > 3].copy()
    negative_reviews = reviews[reviews['sentiment'] < 3].copy()

    # Save positive and negative reviews to CSV files
    positive_reviews.to_csv('positive.csv', index=False)
    negative_reviews.to_csv('negative.csv', index=False)

    # Return the positive reviews, negative reviews, and average sentiment score
    return positive_reviews, negative_reviews, sentiment_avg
