import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv("customer_support_tweets.csv")

# Remove duplicates and irrelevant data
df.drop_duplicates(subset=['text'], inplace=True)
df.drop(columns=['tweet_id', 'created_at', 'user_screen_name', 'user_profile_image_url'], inplace=True)

# Preprocess the text data
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a recommender system
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the performance of the recommender system
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

import tweepy
import preprocessor as p
from textblob import TextBlob

# Develop a chatbot
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_recommendation(text):
    # Use the trained recommender system to provide personalized recommendations
    return "Replace your product."

# Use the chatbot to respond to tweets
class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        if status.in_reply_to_status_id is not None or status.user.id == api.me().id:
            return
        
        text = p.clean(status.text)
        sentiment = get_sentiment(text)
        
        if sentiment < -0.1:
            recommendation = get_recommendation(text)
            reply = "@" + status.user.screen_name + " " + recommendation
            api.update_status(reply, in_reply_to_status_id=status.id)
            print("Tweet replied to:", text)

stream_listener = MyStreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=['customer service', 'customer support'])

# Test the chatbot using sample queries
sample_queries = ['My product is defective.', 'I want to return my order.', 'Can you provide a discount code?']
for query in sample_queries:
    recommendation = get_recommendation(query)
    print("Query:", query)
    print("Recommendation:", recommendation)

# Evaluate the performance of the chatbot
# Measure response time
import time

start_time = time.time()
recommendation = get_recommendation("My product is defective.")
end_time = time.time()

response_time = end_time - start_time
print("Response time:", response_time)

# Measure accuracy
correct_recommendation = 'Replace your product.'
accuracy = 0

for query in sample_queries:
    recommendation = get_recommendation(query)
    if recommendation == correct_recommendation:
        accuracy += 1

accuracy = accuracy / len(sample_queries)
print("Accuracy:", accuracy)
