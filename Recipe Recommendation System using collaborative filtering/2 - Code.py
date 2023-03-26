import pandas as pd

def load_data(file_path):
    """
    Load data from the given file path and return as pandas DataFrame.
    """
    data = pd.read_json(file_path)
    return data


from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """
    Preprocess the given data by converting categorical variables to numerical using Label Encoding.
    """
    le = LabelEncoder()
    data['encoded_cuisine'] = le.fit_transform(data['cuisine'])
    return data


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def train_model(data):
    """
    Train the Collaborative Filtering model using SVD algorithm on the given data.
    """
    reader = Reader(rating_scale=(0, 1))
    surprise_data = Dataset.load_from_df(data[['id', 'encoded_cuisine', 'ingredient_list']], reader)
    trainset, testset = train_test_split(surprise_data, test_size=0.2)
    algo = SVD(n_factors=100, n_epochs=20, biased=True)
    algo.fit(trainset)
    return algo, testset


from surprise import accuracy

def evaluate_model(algo, testset):
    """
    Evaluate the Collaborative Filtering model on the given testset.
    """
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    return rmse, mae


def predict_recipe(algo, data, recipe_id):
    """
    Predict the cuisine for the given recipe ID using the trained Collaborative Filtering model.
    """
    recipe_index = data[data['id'] == recipe_id].index[0]
    recipe_predictions = algo.predict(recipe_index, verbose=False)
    predicted_cuisine = data[data['encoded_cuisine'] == recipe_predictions.est].iloc[0]['cuisine']
    return predicted_cuisine
