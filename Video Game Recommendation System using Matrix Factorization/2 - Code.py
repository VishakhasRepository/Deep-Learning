# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import SVD, accuracy, Dataset, Reader
from surprise.model_selection import GridSearchCV
import joblib

# Section 1: Data preprocessing
# Load and clean the dataset
df = pd.read_csv('video_games.csv')
df.dropna(inplace=True)

# Transform the dataset to a suitable format for matrix factorization
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Section 2: Matrix factorization
# Apply Singular Value Decomposition (SVD) to decompose the user-item rating matrix
param_grid = {'n_epochs': [20, 25], 'lr_all': [0.005, 0.01],
              'reg_all': [0.4, 0.6]}
gs_model = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs_model.fit(data)
best_model = gs_model.best_estimator['rmse']

# Obtain user and item factors
trainset = data.build_full_trainset()
best_model.fit(trainset)

# Section 3: Model evaluation
# Evaluate the performance of the matrix factorization models using appropriate metrics
predictions = best_model.test(testset)
print("RMSE:", accuracy.rmse(predictions))
print("MAE:", accuracy.mae(predictions))

# Compare the performance of different matrix factorization models
models = [SVD(n_epochs=20, lr_all=0.005, reg_all=0.4),
          SVD(n_epochs=20, lr_all=0.01, reg_all=0.4),
          SVD(n_epochs=25, lr_all=0.005, reg_all=0.4),
          SVD(n_epochs=25, lr_all=0.01, reg_all=0.4),
          SVD(n_epochs=20, lr_all=0.005, reg_all=0.6),
          SVD(n_epochs=20, lr_all=0.01, reg_all=0.6),
          SVD(n_epochs=25, lr_all=0.005, reg_all=0.6),
          SVD(n_epochs=25, lr_all=0.01, reg_all=0.6)]
results = []
for model in models:
    results.append(accuracy.rmse(model.fit(trainset).test(testset)))

# Section 4: Recommendation generation
# Generate recommendations for video games based on user preferences and the learned user and item factors
joblib.dump(best_model, 'model.pkl')
model = joblib.load('model.pkl')
user_id = 1234
items_to_recommend = []
for item_id in df['product_id'].unique():
    predicted_rating = model.predict(user_id, item_id)[3]
    if predicted_rating > 4:
        items_to_recommend.append(item_id)
print(items_to_recommend)

# Evaluate the quality of the recommendations using appropriate metrics
# Implement the recommendation system in a web application

# Section 5: Deployment
# Deploy the recommendation system on a cloud platform such as AWS or GCP
# Ensure scalability and reliability of the system
# Monitor and optimize the system's performance

# Section 6: Future work
# Explore advanced matrix factorization techniques such
