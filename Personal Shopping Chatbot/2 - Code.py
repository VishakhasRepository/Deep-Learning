import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
df = pd.read_csv('train.csv')

# Remove irrelevant features
df = df.drop(['id', 'product_uid', 'relevance'], axis=1)

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Perform text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

df['title'] = df['product_title'].apply(lambda x: ' '.join([stemmer.stem(w.lower()) for w in word_tokenize(x) if w.lower() not in stop_words]))
df['description'] = df['product_description'].apply(lambda x: ' '.join([stemmer.stem(w.lower()) for w in word_tokenize(x) if w.lower() not in stop_words]))

# Perform feature engineering
df['brand'] = df['product_title'].apply(lambda x: x.split()[0])
df['color'] = df['product_title'].apply(lambda x: x.split()[-1])
df['material'] = df['product_description'].apply(lambda x: ' '.join([w for w in word_tokenize(x) if 'material' in w]))

# Save the cleaned dataset
df.to_csv('cleaned_data.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Analyze relevance scores
plt.hist(df['relevance'])
plt.show()

# Explore relationships between features
plt.scatter(df['brand'], df['relevance'])
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['title', 'description', 'brand', 'color', 'material']], df['relevance'], test_size=0.2, random_state=42)

# Train and evaluate a random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['title', 'description', 'brand', 'color', 'material']], df['relevance'], test_size=0.2, random_state=42)

# Train and evaluate a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Perform hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Best Parameters:', best_params)

# Train and evaluate the model with the best hyperparameters
best_clf = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision (best model):', precision)
print('Recall (best model):', recall)
print('F1 Score (best model):', f1)
