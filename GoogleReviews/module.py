import googlemaps
import pandas as pd

def get_Google_Reviews(place_name):
    # Initialize the Google Maps client with the provided API key
    gmaps = googlemaps.Client(key='AIzaSyCcyByaOJSV_VoczzbEUGnKwibLc4SFC8w')
    
    # Search for the place using the provided place name
    places_result = gmaps.places(place_name)
    
    # Extract the place ID of the first result
    place_id = places_result['results'][0]['place_id']
    
    # Retrieve detailed information about the place using the place ID
    place = gmaps.place(place_id=place_id)
    
    # Initialize an empty list to store reviews
    reviews = []
    
    # Loop through the reviews and extract the text of each review
    for i in range(len(place['result']['reviews'])):
        text = place['result']['reviews'][i]['text']
        reviews.append({'text': text})
    
    # Convert the list of reviews into a DataFrame
    df = pd.DataFrame(reviews)
    
    # Return the DataFrame containing the reviews
    return df
