import requests
from bs4 import BeautifulSoup

# URL of CNN website
url = "https://www.cnn.com"

# Make a request to the URL
response = requests.get(url)

# Create a BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# Find all the article links on the page
articles = soup.find_all("a", class_="news__title")

# Loop through the article links and print the title and URL of each article
for article in articles:
    print(article.text)
    print(article['href'])

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Define a list of stop words
stop_words = set(stopwords.words('english'))

# Define a function to clean the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words and punctuations
    clean_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    # Join the tokens back into a string
    clean_text = " ".join(clean_tokens)
    return clean_text

# Example usage
text = "This is an example text! It contains stop words like 'the', 'and', and 'a'."
clean_text = clean_text(text)
print(clean_text)

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')

# Define a function to extract keywords from the text
def extract_keywords(text, num_keywords=5):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Calculate the frequency distribution of the tokens
    freq_dist = FreqDist(tokens)
    # Get the most common tokens as keywords
    keywords = freq_dist.most_common(num_keywords)
    return keywords

# Example usage
text = "This is an example text! It contains keywords like 'example', 'text', and 'keywords'."
keywords = extract_keywords(text)
print(keywords)

# Machine Learning Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Vectorize the text data using count vectorizer
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(df_train['text'])

# Train the model using Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(X_train_counts, df_train['category'])

# Evaluate the model using test data
X_test_counts = count_vect.transform(df_test['text'])
predicted = clf.predict(X_test_counts)

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(df_test['category'], predicted))

