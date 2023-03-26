# Import the necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Load the dataset into a pandas dataframe
df = pd.read_csv('UNSW-NB15.csv')

# Preprocessing: drop unnecessary columns and encode categorical variables
df = df.drop(['id', 'proto', 'service', 'state'], axis=1)
df = pd.get_dummies(df)

# Split the dataset into training and testing sets
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest pipeline with hyperparameter tuning and feature selection
pipe_rf = Pipeline([
    ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'feature_selection__threshold': ['mean', 'median', '1.25*median'],
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [10, 20, 30],
    'clf__max_features': ['sqrt', 'log2']
}

# Use GridSearchCV to find the best hyperparameters for the Random Forest pipeline
grid_search = GridSearchCV(pipe_rf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the best cross-validation score
print('Best Hyperparameters:', grid_search.best_params_)
print('Best Cross-validation Score:', grid_search.best_score_)

# Evaluate the performance of the best estimator using cross-validation
rf_best = grid_search.best_estimator_
scores = cross_val_score(rf_best, X, y, cv=10, n_jobs=-1)
print('Cross-validation Scores:', scores)
print('Mean Cross-validation Score:', scores.mean())

# Evaluate the performance of the best estimator on the test set
y_pred = rf_best.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))
