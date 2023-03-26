# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('online_shopping.csv')

# Preprocess data
# Some preprocessing steps like removing duplicate entries, cleaning up missing values etc.

# Create user-item matrix
user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Define a function to calculate similarity
def similarity_matrix(data_matrix):
    """
    Calculate the similarity between all items in the dataset.
    """
    return cosine_similarity(data_matrix.T)

# Calculate item-item similarity
item_similarity = similarity_matrix(user_item_matrix)

# Define a function to make recommendations
def get_recommendations(item_id, user_item_matrix, item_similarity, n_recommendations=10):
    """
    Return top n recommended items for a given item based on user-item ratings and item-item similarity.
    """
    # Get user-item ratings for the given item
    item_ratings = user_item_matrix[item_id]
    
    # Calculate weighted average similarity score for all items based on item similarity and user-item ratings
    weighted_item_ratings = item_ratings * item_similarity
    weighted_item_ratings = weighted_item_ratings.sum(axis=1) / item_similarity.sum(axis=1)
    
    # Sort items by weighted average similarity score and return top n items
    recommended_items = weighted_item_ratings.sort_values(ascending=False)[:n_recommendations]
    
    return recommended_items.index.tolist()

# Example usage to get top 5 recommended items for item 123
get_recommendations(123, user_item_matrix, item_similarity, n_recommendations=5)
