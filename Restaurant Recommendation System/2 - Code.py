import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv('ratings.csv')
restaurants = pd.read_csv('restaurants.csv')

# Merge ratings and restaurants dataframes
df = pd.merge(ratings, restaurants, on='restaurant_id')

# Pivot table to transform data into a user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='restaurant_name', values='rating')

# Replace missing values with 0
user_item_matrix.fillna(0, inplace=True)

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(user_item_matrix)

def get_similar_users(user_id, k=5):
    """
    Returns top k similar users based on cosine similarity.
    """
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = cosine_sim[user_idx].argsort()[::-1][1:k+1]
    return user_item_matrix.iloc[similar_users].index.tolist()

def recommend_restaurants(user_id, n=5):
    """
    Recommends top n restaurants for a user based on user-based collaborative filtering.
    """
    similar_users = get_similar_users(user_id)
    user_ratings = user_item_matrix.loc[user_id]
    similar_user_ratings = user_item_matrix.loc[similar_users]
    # Calculate weighted average of ratings
    recommendations = similar_user_ratings.apply(lambda x: np.dot(x, user_ratings) / x.sum(), axis=1)
    # Remove restaurants that user has already rated
    recommendations = recommendations[user_ratings == 0]
    # Sort by rating and return top n restaurants
    return recommendations.sort_values(ascending=False).head(n).index.tolist()

# Calculate item-item similarity matrix
item_item_sim = cosine_similarity(user_item_matrix.T)

def get_similar_items(item_name, k=5):
    """
    Returns top k similar items based on cosine similarity.
    """
    item_idx = user_item_matrix.columns.get_loc(item_name)
    similar_items = item_item_sim[item_idx].argsort()[::-1][1:k+1]
    return user_item_matrix.columns[similar_items].tolist()

def recommend_restaurants_item_based(user_id, n=5):
    """
    Recommends top n restaurants for a user based on item-based collaborative filtering.
    """
    user_ratings = user_item_matrix.loc[user_id]
    # Get similar items for restaurants that user has rated
    similar_items = user_ratings[user_ratings != 0].index.map(lambda x: get_similar_items(x))
    # Combine similar items and remove duplicates
    similar_items = pd.Series([item for sublist in similar_items for item in sublist]).drop_duplicates()
    # Remove restaurants that user has already rated
    recommendations = similar_items[~similar_items.isin(user_ratings.index)]
    # Calculate weighted average of ratings
    recommendations = recommendations.apply(lambda x: np.dot(user_ratings, user_item_matrix[x]) / user_item_matrix[x].sum())
    # Sort by rating and return top n restaurants
    return recommendations.sort_values(ascending=False).head(n).index.tolist()

# Get user input for their preferred cuisine and location
cuisine_input = input("What type of cuisine are you in the mood for? ")
location_input = input("Where are you located? ")

# Get recommended restaurants based on user input
recommendations = get_recommendations(cuisine_input, location_input, restaurants_df, reviews_df)

# Print out the top 5 recommended restaurants
print("Here are the top 5 recommended restaurants for you:")
for i, restaurant in recommendations.head().iterrows():
    print(f"{i + 1}. {restaurant['name']} - {restaurant['address']}, {restaurant['city']}")
    print(f"Rating: {restaurant['rating']} ({restaurant['review_count']} reviews)")
    print(f"Price Range: {restaurant['price_range']}")
    print(f"Cuisine: {restaurant['cuisine']}")
    print()
