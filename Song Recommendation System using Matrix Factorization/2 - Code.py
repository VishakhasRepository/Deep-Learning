import pandas as pd
import numpy as np

# Load data
songs_df = pd.read_csv('songs.csv')
interactions_df = pd.read_csv('interactions.csv')

# Data preprocessing
songs_df['song_id'] = songs_df.index
interactions_df = interactions_df.merge(songs_df[['song_id', 'song_name']], on='song_id')
interactions_df['interaction'] = 1

from sklearn.model_selection import train_test_split

# Train-test split
train_data, test_data = train_test_split(interactions_df, test_size=0.2, random_state=42)

# Create user-item matrix
user_item_matrix = train_data.pivot_table(index='user_id', columns='song_name', values='interaction')
user_item_matrix = user_item_matrix.fillna(0)

import implicit

# Training the model
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
model.fit(user_item_matrix.T)

from sklearn.metrics import roc_auc_score

# Predict on test data
test_user_item_matrix = test_data.pivot_table(index='user_id', columns='song_name', values='interaction').fillna(0)
test_users = test_user_item_matrix.index

# Evaluate the model
auc_scores = []
for user in test_users:
    user_auc_score = roc_auc_score(test_user_item_matrix.loc[user], model.recommend(user, user_item_matrix, N=10))
    auc_scores.append(user_auc_score)

mean_auc_score = np.mean(auc_scores)
print(f"Mean AUC Score: {mean_auc_score}")

user_id = 1

# Get the user-item matrix
user_item_matrix = interactions_df.pivot_table(index='user_id', columns='song_name', values='interaction')
user_item_matrix = user_item_matrix.fillna(0)

# Get the user's interactions
user_interactions = interactions_df[interactions_df['user_id'] == user_id]

# Get the user's recommendations
user_recommendations = model.recommend(user_id, user_item_matrix, N=10)
user_recommendations = pd.DataFrame(user_recommendations, columns=['song_name', 'score'])

# Print the user's recommendations
print(f"Recommendations for User {user_id}:")
for idx, row in user_recommendations.iterrows():
    if row['song_name'] not in user_interactions['song_name'].tolist():
        print(f"{row['song_name']} with score {row['score']}")
