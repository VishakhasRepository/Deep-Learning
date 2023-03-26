import pandas as pd

# Load the dataset
books_df = pd.read_csv('books.csv', error_bad_lines=False)

# Preview the dataset
print(books_df.head())
print('\n')

# Check for missing values
print(books_df.isnull().sum())
print('\n')

# Check for duplicates
print(books_df.duplicated().sum())
print('\n')

import re

# Remove unwanted columns
books_df.drop(['bookID', 'isbn', 'isbn13'], axis=1, inplace=True)

# Remove duplicates
books_df.drop_duplicates(subset=['title', 'authors'], keep='first', inplace=True)

# Fill missing values with unknown
books_df.fillna('unknown', inplace=True)

# Extract year from publication date
books_df['year'] = books_df['publication_date'].apply(lambda x: re.findall(r'\d{4}', str(x))[0] if re.findall(r'\d{4}', str(x)) else 'unknown')

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the title and authors columns
books_df['title_vector'] = tfidf_vectorizer.fit_transform(books_df['title'])
books_df['authors_vector'] = tfidf_vectorizer.fit_transform(books_df['authors'])

from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
title_similarity = cosine_similarity(books_df['title_vector'])
author_similarity = cosine_similarity(books_df['authors_vector'])

# Combine the similarity matrices
similarity_matrix = (title_similarity + author_similarity) / 2

import numpy as np

# Get the index of the input book
index = books_df[books_df['title'] == 'The Hobbit'].index[0]

# Get the pairwise similarity scores of the input book
similarity_scores = list(enumerate(similarity_matrix[index]))

# Sort the similarity scores in descending order
sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Get the top 10 most similar books
top_books = sorted_scores[1:11]

# Print the recommended books
print("Recommended Books:")
for book in top_books:
    print(books_df.iloc[book[0]]['title'])
