import googlemaps
import pandas as pd

def get_Google_Reviews(place_name):
    gmaps = googlemaps.Client(key='AIzaSyCcyByaOJSV_VoczzbEUGnKwibLc4SFC8w')
    # place_name = '正月初五手作鮮肉湯包'

    places_result = gmaps.places(place_name)
    place_id = places_result['results'][0]['place_id']
    place = gmaps.place(place_id = place_id)
    reviews = []
    for i in range(len(place['result']['reviews'])):
        text = place['result']['reviews'][i]['text']
        reviews.append({'text':text})

    df = pd.DataFrame(reviews)
    return df

