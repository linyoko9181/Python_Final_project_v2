from flask import Flask, render_template, request, redirect, url_for, session
from GoogleReviews.module import get_Google_Reviews
from TextSelection.module import text_selector
from SentimentAnalysis.module import sentiment_analyzer
from DataExtraction.module import data_extractor

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route("/", methods=['GET','POST'])
def restaurant():
    if request.method == "POST":
        name = request.form['name']
        df = get_Google_Reviews(name)
        positive_reviews, negative_reviews, rating = sentiment_analyzer(df)
        positive_labels, negative_labels = data_extractor(positive_reviews,negative_reviews,name)

        session['positive_labels'] = positive_labels
        session['negative_labels'] = negative_labels
        session['name'] = name
        session['rating'] = rating
        return redirect(url_for('sentiment'))
    else:
        return render_template('index.html')
    
@app.route("/sentiment", methods=['GET'])
def sentiment():
    rating = session.get("rating")
    restaurant_name = session.get('name', '')
    return render_template('sentiment.html', restaurant_name=restaurant_name, rating = rating)

@app.route("/positive", methods=['GET', 'POST'])
def show_positive_review():
    positive_labels = session.get('positive_labels', [])
    rating = session.get("rating")
    if request.method == "POST":
        label = request.form['Label']
        displayed_text = text_selector(1, label)
    else:
        displayed_text = text_selector(1, positive_labels[1][0])
    return render_template('positive.html', labels=positive_labels, displayed_text=displayed_text, rating = rating)

@app.route("/negative", methods=['GET','POST'])
def show_negative_review():
    negative_labels = session.get('negative_labels', [])
    rating = session.get("rating")
    if request.method == "POST":
        label = request.form['Label']
        displayed_text = text_selector(2, label)
    else:
        displayed_text = text_selector(2, negative_labels[1][0])
    return render_template('negative.html', labels=negative_labels, displayed_text=displayed_text, rating = rating)

if __name__ == "__main__":
    app.run(debug=True)
