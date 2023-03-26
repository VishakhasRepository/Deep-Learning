# Section 1: Importing the necessary libraries and dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load the dataset from a CSV file
data = pd.read_csv('insurance_fraud.csv')

# Section 2: Data Preprocessing

# Drop rows with missing values
data.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data)

# Split the data into training and testing sets
X = data.drop(['fraud_reported_Y'], axis=1)
y = data['fraud_reported_Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Section 3: Model Training

# Initialize a random forest classifier with default hyperparameters
clf = RandomForestClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Section 4: Model Evaluation

# Predict the fraud label for the test data
y_pred = clf.predict(X_test)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
