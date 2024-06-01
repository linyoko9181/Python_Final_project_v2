from flask import Flask, render_template, request, redirect, url_for, session
from GoogleReviews.module import get_Google_Reviews
from TextSelection.module import text_selector
from SentimentAnalysis.module import sentiment_analyzer
from DataExtraction.module import data_extractor

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Route for the homepage where users input restaurant name
@app.route("/", methods=['GET','POST'])
def restaurant():
    if request.method == "POST":
        # Retrieve restaurant name from the form
        name = request.form['name']
        # Get Google reviews for the provided restaurant name
        df = get_Google_Reviews(name)
        # Analyze sentiment of the reviews
        positive_reviews, negative_reviews, rating = sentiment_analyzer(df)
        # Extract relevant data from positive and negative reviews
        positive_labels, negative_labels = data_extractor(positive_reviews, negative_reviews, name)

        # Store data in session for later use
        session['positive_labels'] = positive_labels
        session['negative_labels'] = negative_labels
        session['name'] = name
        session['rating'] = rating
        # Redirect to the sentiment analysis results page
        return redirect(url_for('sentiment'))
    else:
        # Render the homepage template
        return render_template('index.html')

# Route for displaying sentiment analysis results
@app.route("/sentiment", methods=['GET'])
def sentiment():
    # Retrieve data from session
    rating = session.get("rating")
    restaurant_name = session.get('name', '')
    # Render the sentiment analysis results template
    return render_template('sentiment.html', restaurant_name=restaurant_name, rating=rating)

# Route for displaying positive reviews
@app.route("/positive", methods=['GET', 'POST'])
def show_positive_review():
    # Retrieve positive labels from session
    positive_labels = session.get('positive_labels', [])
    rating = session.get("rating")
    if request.method == "POST":
        # Get selected label from the form
        label = request.form['Label']
        # Retrieve and display text corresponding to the selected label
        displayed_text = text_selector(1, label)
    else:
        # Default to displaying text for the first positive label
        displayed_text = text_selector(1, positive_labels[1][0])
    # Render the positive reviews template
    return render_template('positive.html', labels=positive_labels, displayed_text=displayed_text, rating=rating)

# Route for displaying negative reviews
@app.route("/negative", methods=['GET','POST'])
def show_negative_review():
    # Retrieve negative labels from session
    negative_labels = session.get('negative_labels', [])
    rating = session.get("rating")
    if request.method == "POST":
        # Get selected label from the form
        label = request.form['Label']
        # Retrieve and display text corresponding to the selected label
        displayed_text = text_selector(2, label)
    else:
        # Default to displaying text for the first negative label
        displayed_text = text_selector(2, negative_labels[1][0])
    # Render the negative reviews template
    return render_template('negative.html', labels=negative_labels, displayed_text=displayed_text, rating=rating)

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
