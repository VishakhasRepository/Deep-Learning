import pandas as pd

# Load the Netflix dataset
df = pd.read_csv('netflix_dataset.csv')

# Clean the data
df = df.dropna()

# Merge tables
show_titles = pd.read_csv('tv_show_titles.csv')
df = pd.merge(df, show_titles, on='show_id')

# Create pivot table of user ratings
pivot = pd.pivot_table(df, values='rating', index='user_id', columns='title', fill_value=0)

from sklearn.neighbors import NearestNeighbors

# Split data into training and testing sets
train_data = pivot.iloc[:500, :500]
test_data = pivot.iloc[500:, :500]

# Fit algorithm on training set
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(train_data)

# Evaluate performance on testing set
distances, indices = knn.kneighbors(test_data, n_neighbors=5)

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'metric': ['cosine', 'euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

# Optimize hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(train_data)
best_knn = grid_search.best_estimator_

import numpy as np

# Generate TV show recommendations for users
user_id = 500
user_ratings = pivot.iloc[user_id]
user_ratings = user_ratings[user_ratings != 0]

# Find similar users
distances, indices = best_knn.kneighbors([user_ratings], n_neighbors=10)
similar_users = train_data.iloc[indices[0]]

# Predict ratings for unrated TV shows
predicted_ratings = []
for title in pivot.columns:
    if user_ratings[title] == 0:
        predicted_rating = np.mean(similar_users[title])
        predicted_ratings.append(predicted_rating)
    else:
        predicted_ratings.append(0)

# Return top recommendations
top_recommendations = pd.Series(predicted_ratings, index=pivot.columns)
top_recommendations = top_recommendations.sort_values(ascending=False)[:10]

from sklearn.metrics import precision_recall_fscore_support

# Evaluate performance of recommendation system
precision, recall, f1_score, _ = precision_recall_fscore_support(test_data.loc[user_id], top_recommendations, average='binary')

print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1_score))
