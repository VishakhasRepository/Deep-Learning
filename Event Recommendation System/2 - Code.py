import pandas as pd
import numpy as np

# Load the event data from various sources
meetup_data = pd.read_csv('meetup_data.csv')
eventbrite_data = pd.read_csv('eventbrite_data.csv')
facebook_data = pd.read_csv('facebook_data.csv')

# Combine the event data from various sources
event_data = pd.concat([meetup_data, eventbrite_data, facebook_data])

# Drop the rows with missing values
event_data = event_data.dropna()

# Drop the unnecessary columns
event_data = event_data.drop(['event_id', 'start_time', 'end_time', 'event_url'], axis=1)

# Convert the date column to datetime format
event_data['date'] = pd.to_datetime(event_data['date'])

# Convert the free event column to boolean format
event_data['free_event'] = event_data['free_event'].map({'Yes': True, 'No': False})

# Create a new column for the month of the event
event_data['month'] = event_data['date'].dt.month

# Encode the categorical columns
event_data = pd.get_dummies(event_data, columns=['category', 'venue_city'])

# Scale the numerical columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
event_data[['attendance', 'price']] = scaler.fit_transform(event_data[['attendance', 'price']])

#Feature Engineering
#Create new features based on the event type, date and time, and location of the events
#Extract date and time information
events['start_time'] = pd.to_datetime(events['start_time'])
events['start_year'] = events['start_time'].dt.year
events['start_month'] = events['start_time'].dt.month
events['start_day'] = events['start_time'].dt.day
events['start_hour'] = events['start_time'].dt.hour
events['start_minute'] = events['start_time'].dt.minute

#Create dummy variables for event type and location
event_type_dummies = pd.get_dummies(events['event_type'], prefix='event_type')
location_dummies = pd.get_dummies(events['location'], prefix='location')
events = pd.concat([events, event_type_dummies, location_dummies], axis=1)

#Step 4: User Profiling
#Profile the users based on their interests, past event history, and demographic information
#Merge events and users on the event_id column
event_attendees = pd.merge(event_attendees, events[['event_id', 'start_year', 'start_month', 'start_day', 'start_hour', 'start_minute', 'event_type', 'location']], on='event_id', how='left')
user_profiles = pd.DataFrame({'user_id': np.unique(event_attendees['user_id'])})
for event_type in np.unique(events['event_type']):
user_profiles[f'num_{event_type}'] = event_attendees[event_attendees['event_type'] == event_type]['user_id'].value_counts()
user_profiles.fillna(0, inplace=True)
user_profiles.set_index('user_id', inplace=True)

#Convert user age to numeric
users['age'] = pd.to_numeric(users['age'], errors='coerce')

#Merge user and event attendee data on the user_id column
user_attendees = pd.merge(event_attendees, users, on='user_id', how='left')
user_profiles['avg_age'] = user_attendees.groupby('user_id')['age'].mean()
user_profiles['num_attended'] = user_attendees.groupby('user_id')['event_id'].nunique()
user_profiles['num_tickets'] = user_attendees.groupby('user_id')['tickets'].sum()
user_profiles.fillna(0, inplace=True)

#Scale the features
scaler = StandardScaler()
user_profiles_scaled = scaler.fit_transform(user_profiles)

# Step 5: Collaborative Filtering
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Convert the dataset to the Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train[['user_id', 'event_id', 'rsvp']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the SVD model
model = SVD(n_factors=20, n_epochs=10, lr_all=0.005, reg_all=0.4)
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Step 6: Content-Based Filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TfidfVectorizer object to compute the TF-IDF scores
tfidf = TfidfVectorizer(stop_words='english')

# Compute the TF-IDF scores for the event descriptions
event_descriptions = data['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(event_descriptions)

# Compute the pairwise cosine similarity between the event descriptions
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build a mapping between event IDs and event indices
indices = pd.Series(data.index, index=data['event_id'])

# Define a function to recommend similar events based on the event description
def content_based_recommendations(event_id, cosine_sim=cosine_sim):
    # Get the index of the event that matches the event ID
    idx = indices[event_id]

    # Get the pairwise cosine similarity scores for the event
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the events based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 3 most similar events
    sim_scores = sim_scores[1:4]

    # Get the event indices for the top 3 most similar events
    event_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar events
    return data.iloc[event_indices]['event_id'].tolist()

# Step 7: Hybrid Recommendation Engine
def hybrid_recommendations(user_id):
    # Get the list of events attended by the user
    attended_events = data[data['user_id'] == user_id]['event_id'].tolist()

    # Get the list of similar events using content-based filtering
    similar_events = []
    for event_id in attended_events:
        similar_events += content_based_recommendations(event_id)

    # Remove duplicates from the list of similar events
    similar_events = list(set(similar_events))

    # Get the predicted ratings for the similar events using collaborative filtering
    event_ratings = []
    for event_id in similar_events:
        event_ratings.append((event_id, model.predict(user_id, event_id)[3]))

    # Sort the list of similar events based on the predicted ratings
    event_ratings = sorted(event_ratings, key=lambda x: x[1], reverse=True)

    # Return the top 3 recommended events
    return [x[0] for x in event_ratings[:3]]

    from sklearn.metrics import accuracy_score, precision_score, recall_score

# evaluate collaborative filtering model
cf_preds = cf_model.predict(test_data[['user_id', 'event_id']])
cf_preds = np.round(cf_preds)
cf_accuracy = accuracy_score(test_data['interested'], cf_preds)
cf_precision = precision_score(test_data['interested'], cf_preds)
cf_recall = recall_score(test_data['interested'], cf_preds)
print("Collaborative Filtering Model:")
print(f"Accuracy: {cf_accuracy:.4f}")
print(f"Precision: {cf_precision:.4f}")
print(f"Recall: {cf_recall:.4f}")

# evaluate content-based filtering model
cb_preds = cb_model.predict(test_data[features])
cb_preds = np.round(cb_preds)
cb_accuracy = accuracy_score(test_data['interested'], cb_preds)
cb_precision = precision_score(test_data['interested'], cb_preds)
cb_recall = recall_score(test_data['interested'], cb_preds)
print("Content-Based Filtering Model:")
print(f"Accuracy: {cb_accuracy:.4f}")
print(f"Precision: {cb_precision:.4f}")
print(f"Recall: {cb_recall:.4f}")

# evaluate hybrid model
hybrid_preds = hybrid_model.predict(test_data[['user_id', 'event_id']]+test_data[features].values.tolist())
hybrid_preds = np.round(hybrid_preds)
hybrid_accuracy = accuracy_score(test_data['interested'], hybrid_preds)
hybrid_precision = precision_score(test_data['interested'], hybrid_preds)
hybrid_recall = recall_score(test_data['interested'], hybrid_preds)
print("Hybrid Filtering Model:")
print(f"Accuracy: {hybrid_accuracy:.4f}")
print(f"Precision: {hybrid_precision:.4f}")
print(f"Recall: {hybrid_recall:.4f}")

# This code evaluates the performance of the collaborative filtering, content-based filtering, and hybrid models using metrics such as accuracy, precision, and recall. The code predicts the interest of users in the test set and compares the predictions with the actual values to calculate the accuracy, precision, and recall of each model. The evaluation of the models helps in determining the model that provides the best recommendations to the users.

# To use this code, make sure to replace the cf_model, cb_model, and hybrid_model variables with the respective models that you have trained. Also, replace the test_data variable with the test dataset that you have prepared.