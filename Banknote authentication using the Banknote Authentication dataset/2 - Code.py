# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
banknote_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", header=None, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

# Splitting the data into features and labels
X = banknote_data.iloc[:, :-1].values
y = banknote_data.iloc[:, -1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating an instance of Logistic Regression classifier
classifier = LogisticRegression()

# Fitting the classifier to the training data
classifier.fit(X_train, y_train)

# Predicting the labels of the test data
y_pred = classifier.predict(X_test)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Displaying the accuracy of the model
print("Accuracy: {:.2f}%".format(accuracy * 100))
