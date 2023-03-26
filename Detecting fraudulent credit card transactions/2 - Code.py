# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Loading and Exploring the Dataset
data = pd.read_csv("creditcard.csv")
print(data.head())
print(data.describe())
print(data.info())
print(data['Class'].value_counts())
sns.countplot(data['Class'])

# Data Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

# Model Selection
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Multi-Layer Perceptron": MLPClassifier(max_iter=1000)
}
parameters = {
    "Logistic Regression": {
        "penalty": ['l1', 'l2', 'elasticnet'],
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    "Decision Tree": {
        "criterion": ['gini', 'entropy'],
        "max_depth": [2, 4, 6, 8, 10, 12]
    },
    "Random Forest": {
        "n_estimators": [100, 500, 1000],
        "criterion": ['gini', 'entropy'],
        "max_depth": [2, 4, 6, 8, 10, 12]
    },
    "Multi-Layer Perceptron": {
        "hidden_layer_sizes": [(50,50,50), (50,100,50), (100,)],
        "activation": ['tanh', 'relu'],
        "solver": ['sgd', 'adam'],
        "alpha": [0.0001, 0.05],
        "learning_rate": ['constant','adaptive']
    }
}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, parameters[model_name], cv=5)
    grid_search.fit(X_train, y_train)
    print(model_name)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    y_pred = grid_search.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Model Interpretation
rfc = RandomForestClassifier(criterion='gini', max_depth=12, n_estimators=1000)
rfc.fit(X_train, y_train)
feature_importance = pd.Series(rfc.feature_importances_, index=data.columns[:-1])
feature_importance.nlargest(10).plot(kind='barh')
plt.show()
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
