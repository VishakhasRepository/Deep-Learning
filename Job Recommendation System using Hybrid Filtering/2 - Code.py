import pandas as pd
import numpy as np

# Load the job seeker and job data from CSV files
job_seeker_data = pd.read_csv('job_seeker_data.csv')
job_data = pd.read_csv('job_data.csv')

# Preprocess the data by removing irrelevant information and cleaning the data
job_seeker_data = job_seeker_data.drop(['Name', 'Email', 'Phone'], axis=1)
job_data = job_data.drop(['Company', 'Location'], axis=1)

# Prepare the data for analysis by converting categorical variables to numeric
job_seeker_data['Experience'] = job_seeker_data['Experience'].replace({'<1': 0.5, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '10+': 11})
job_seeker_data['Education'] = job_seeker_data['Education'].replace({'High School': 1, 'Associate Degree': 2, 'Bachelor Degree': 3, 'Master Degree': 4, 'PhD': 5})
job_data['Salary'] = job_data['Salary'].replace({'High': 3, 'Medium': 2, 'Low': 1})

from sklearn.metrics.pairwise import cosine_similarity

# Compute the user-item matrix
user_item_matrix = job_seeker_data.pivot_table(index='Job Seeker ID', columns='Job ID', values='Rating')

# Compute the cosine similarity between job seekers
user_similarity = cosine_similarity(user_item_matrix)

# Generate personalized job recommendations for a specific job seeker
job_seeker_id = 123
job_seeker_ratings = user_item_matrix.loc[job_seeker_id, :]
similar_users = user_similarity[job_seeker_id, :].argsort()[::-1][:10]
similar_user_ratings = user_item_matrix.loc[similar_users, :]
job_recommendations = (similar_user_ratings * user_similarity[job_seeker_id, similar_users][:, np.newaxis]).sum(axis=0) / user_similarity[job_seeker_id, similar_users].sum()
job_recommendations = job_recommendations.sort_values(ascending=False)[:10]
 
 from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert the job descriptions and job seeker skills to TF-IDF vectors
job_descriptions = job_data['Description'].tolist()
job_seeker_skills = job_seeker_data['Skills'].tolist()
vectorizer = TfidfVectorizer()
job_description_vectors = vectorizer.fit_transform(job_descriptions)
job_seeker_skill_vectors = vectorizer.transform(job_seeker_skills)

# Compute the cosine similarity between job descriptions and job seeker skills
similarity_matrix = cosine_similarity(job_seeker_skill_vectors, job_description_vectors)

# Generate personalized job recommendations for a specific job seeker
job_seeker_id = 123
job_seeker_similarity = similarity_matrix[job_seeker_id, :]
job_recommendations = job_data.iloc[job_seeker_similarity.argsort()[::-1][:10]]

# Create a Hybrid Recommender System
class HybridRecommender:
    
    def __init__(self, collaborative_filtering_model, content_based_model, ratings_matrix):
        self.cf_model = collaborative_filtering_model
        self.cb_model = content_based_model
        self.ratings_matrix = ratings_matrix
        
    def recommend_items(self, user_id, top_n=5):
        # Get collaborative filtering recommendations
        cf_recs = self.cf_model.recommend_items(user_id, top_n)
        
        # Get content-based recommendations
        rated_items = self.ratings_matrix.loc[user_id].dropna().index.tolist()
        cb_recs = self.cb_model.recommend_items(user_id, top_n+len(rated_items))
        cb_recs = [item for item in cb_recs if item[0] not in rated_items][:top_n]
        
        # Combine the two sets of recommendations
        hybrid_recs = []
        for cf_rec in cf_recs:
            for cb_rec in cb_recs:
                if cf_rec[0] == cb_rec[0]:
                    hybrid_recs.append((cf_rec[0], cf_rec[1]+cb_rec[1]))
        hybrid_recs = sorted(hybrid_recs, key=lambda x: x[1], reverse=True)[:top_n]
        
        return hybrid_recs

# Evaluate the performance of the model
def evaluate_model(model, test_data, users_to_recommend, topn=10):
    # Organize test data by users
    grouped_test_data = test_data.groupby('PersonID')
    
    # Create an empty list to store the AUC scores for each user
    AUC_scores = []
    
    # Iterate over each user and calculate the AUC score
    for person_id, user_data in grouped_test_data:
        # Split the user data into training and test sets
        train_data = user_data['JobID']
        test_data = list(set(users_to_recommend) - set(train_data))
        
        # Create a list of the user's actual job choices
        actual_jobs = [1 if i in train_data else 0 for i in users_to_recommend]
        
        # Generate job recommendations using the model
        job_scores = [model.predict(person_id, i).est for i in users_to_recommend]
        
        # Create a list of the recommended job scores
        recommended_jobs = [1 if i in job_scores else 0 for i in users_to_recommend]
        
        # Calculate the AUC score for this user
        auc_score = roc_auc_score(actual_jobs, recommended_jobs)
        AUC_scores.append(auc_score)
    
    # Calculate the average AUC score for all users
    avg_auc_score = sum(AUC_scores) / len(AUC_scores)
    
    # Print the average AUC score
    print(f"Average AUC Score: {avg_auc_score}")
    
    # Generate top-n job recommendations for each user in the test data
    recommendations = generate_recommendations(model, test_data['PersonID'].unique(), users_to_recommend, topn)
    
    # Calculate the precision and recall scores for the model
    precision = precision_at_k(recommendations, test_data, topn)
    recall = recall_at_k(recommendations, test_data, topn)
    
    # Print the precision and recall scores
    print(f"Precision@{topn}: {precision}")
    print(f"Recall@{topn}: {recall}")

# Evaluate the performance of the model
evaluate_model(hybrid_model, test_data, jobs_data['JobID'].unique(), topn=10)

from flask import Flask, request, jsonify

# Create an instance of the Flask class
app = Flask(__name__)

# Define a function to generate job recommendations for a given user
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Get the user ID from the request
    user_id = request.json['user_id']
    
    # Generate job recommendations for the user
    recommendations = generate_recommendations(hybrid_model, [user_id], jobs_data['JobID'].unique(), topn=10)
    
    # Return the job recommendations as a JSON object
    return jsonify({'recommendations': recommendations})

# Run the Flask app
if __name__ == '__main__':
    app.run()
