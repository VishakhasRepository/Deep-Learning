import pandas as pd

# load the job listings dataset
job_data = pd.read_csv('monster_com-job_sample.csv')

# select relevant columns
job_data = job_data[['job_title', 'job_description', 'location', 'date_added']]

# remove duplicates
job_data = job_data.drop_duplicates(subset=['job_title', 'job_description'])

# remove rows with missing values
job_data = job_data.dropna()

# clean the job descriptions
job_data['job_description'] = job_data['job_description'].str.replace('\n', ' ')
job_data['job_description'] = job_data['job_description'].str.replace('\r', ' ')

# save the cleaned data
job_data.to_csv('cleaned_job_data.csv', index=False)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# load the cleaned job data
job_data = pd.read_csv('cleaned_job_data.csv')

# tokenize the job descriptions
tagged_data = [TaggedDocument(words=word_tokenize(desc.lower()), tags=[str(i)]) for i, desc in enumerate(job_data['job_description'])]

# train a Doc2Vec model
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

# save the trained model
model.save('job_model.doc2vec')


from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

# load the trained Doc2Vec model
model = Doc2Vec.load('job_model.doc2vec')

# get the vectors for all job descriptions
vectors = [model.docvecs[str(i)] for i in range(len(model.docvecs))]

# calculate cosine similarities between job descriptions
similarity_matrix = cosine_similarity(vectors)

# get job recommendations for a given job title
def get_job_recommendations(job_title, n_recommendations=5):
    # get index of job with given title
    job_index = job_data.index[job_data['job_title'] == job_title].tolist()[0]
    
    # get cosine similarities for all jobs compared to the given job
    similarities = similarity_matrix[job_index]
    
    # get indices of top n most similar jobs
    top_indices = similarities.argsort()[-n_recommendations-1:-1][::-1]
    
    # return recommended job titles and descriptions
    recommendations = [(job_data.iloc[index]['job_title'], job_data.iloc[index]['job_description']) for index in top_indices]
    return recommendations


import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the data
data = pd.read_csv("monster_com-job_sample.csv")
jobs = data["job_title"]
descriptions = data["job_description"]

# initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# construct the tf-idf matrix
tfidf_matrix = vectorizer.fit_transform(descriptions)

# get the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# define a function to get a response from the chatbot
def get_response(user_input):
    # convert the user input to lowercase
    user_input = user_input.lower()
    # add the user input to the job descriptions
    descriptions_with_user_input = descriptions.copy()
    descriptions_with_user_input = descriptions_with_user_input.append(pd.Series(user_input))
    # construct the tf-idf matrix for the job descriptions with the user input
    tfidf_matrix_with_user_input = vectorizer.fit_transform(descriptions_with_user_input)
    # get the cosine similarity matrix for the job descriptions with the user input
    cosine_sim_with_user_input = cosine_similarity(tfidf_matrix_with_user_input)
    # get the index of the most similar job
    similar_jobs_indices = cosine_sim_with_user_input[-1].argsort()[:-2:-1]
    # get the job title of the most similar job
    similar_job_title = jobs.iloc[similar_jobs_indices].values[0]
    # generate a response
    if cosine_sim_with_user_input[-1][similar_jobs_indices] == 0:
        response = "I'm sorry, I couldn't find any jobs matching your query."
    else:
        response = f"The best matching job for your query is: {similar_job_title}"
    return response

# define a function to start the chatbot
def start_chatbot():
    print("Hello! I am a job search chatbot. How can I help you?")
    # keep the chatbot running until the user types "bye"
    while True:
        user_input = input("You: ")
        # check if the user wants to quit
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye!")
            break
        # generate a response
        response = get_response(user_input)
        print("Chatbot:", response)

# start the chatbot
start_chatbot()
