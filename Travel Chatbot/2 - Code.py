import concurrent.futures
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define a function to scrape attraction data from a given TripAdvisor URL
def scrape_attraction_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    attraction_name = []
    attraction_rating = []
    for attraction in soup.find_all('div', {'class': 'attraction_clarity_cell'}):
        name = attraction.find('div', {'class': 'listing_title'}).text.strip()
        rating = attraction.find('div', {'class': 'listing_rating'}).find('span')['class'][1].split('_')[1]
        attraction_name.append(name)
        attraction_rating.append(rating)
    return pd.DataFrame({'Name': attraction_name, 'Rating': attraction_rating})

# Define a list of TripAdvisor URLs to scrape data from
urls = ['https://www.tripadvisor.com/Attractions-g60763-Activities-New_York_City_New_York.html',
        'https://www.tripadvisor.com/Attractions-g295424-Activities-Amman_Amman_Governorate.html']

# Scrape attraction data from all URLs using multithreading
with concurrent.futures.ThreadPoolExecutor() as executor:
    attraction_dataframes = list(executor.map(scrape_attraction_data, urls))

# Concatenate all dataframes into a single dataframe
attraction_data = pd.concat(attraction_dataframes)

# Export the dataframe as a CSV file
attraction_data.to_csv('attraction_data.csv', index=False)

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the attraction data
attraction_data = pd.read_csv('attraction_data.csv')

# Define a function to perform text cleaning and lemmatization on attraction names
def clean_attraction_names(attraction_names):
    # Remove non-alphanumeric characters and lowercase all text
    attraction_names = [re.sub(r'[^a-zA-Z0-9\s]', '', name).lower() for name in attraction_names]
    # Tokenize the text
    attraction_names = [word_tokenize(name) for name in attraction_names]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    attraction_names = [[word for word in name if word not in stop_words] for name in attraction_names]
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    attraction_names = [[lemmatizer.lemmatize(word) for word in name] for name in attraction_names]
    # Convert the list of tokenized words back into a string
    attraction_names = [' '.join(name) for name in attraction_names]
    return attraction_names

# Clean the attraction names
attraction_data['Name_clean'] = clean_attraction_names(attraction_data['Name'])

# Export the cleaned data as a CSV file
attraction_data.to_csv('attraction_data_clean.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# load the data
data = pd.read_csv('travel_data.csv')

# create a count vectorizer
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(data['description'])

# compute the cosine similarity matrix based on the count matrix
cosine_sim = cosine_similarity(count_matrix)

# define a function to get recommendations
def get_recommendations(name, cosine_sim=cosine_sim, data=data):
    # find the index of the name in the data
    idx = data[data['name'] == name].index[0]
    
    # get the similarity scores of the name with all other names
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # sort the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # get the indices of the top 5 similar names
    top_indices = [i[0] for i in sim_scores[1:6]]
    
    # return the top 5 similar names
    return data['name'].iloc[top_indices]

# example usage
get_recommendations('Paris')
