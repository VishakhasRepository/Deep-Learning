import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Read the dataset
df = pd.read_csv('language_learning_dataset.csv')

# Remove irrelevant columns
df.drop(['ID', 'Author', 'Date'], axis=1, inplace=True)

# Preprocess the text data
def preprocess(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    text = ' '.join(SnowballStemmer('english').stem(word) for word in text.split())
    return text

df['Text'] = df['Text'].apply(preprocess)

# Extract relevant features
df['Keywords'] = df['Text'].apply(lambda text: ' '.join(set(text.split()[:3])))
df['Topic'] = df['Text'].apply(lambda text: text.split()[3])
df['Difficulty'] = df['Text'].apply(lambda text: text.split()[-1])

# Save the preprocessed dataset
df.to_csv('preprocessed_language_learning_dataset.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_language_learning_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Keywords', 'Topic', 'Difficulty']], df['Resource'], test_size=0.2, random_state=42)

# Train a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the performance of the classifier
y_pred = rfc.predict(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print('Precision:', precision)
print('Recall:', recall)
print('Accuracy:', accuracy)

import random
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_language_learning_dataset.csv')

# Load the trained classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(df[['Keywords', 'Topic', 'Difficulty']], df['Resource'])

# Encode the resource labels
le = LabelEncoder()
le.fit(df['Resource'])

# Define a function to lemmatize text
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens)

# Define a function to get the part of speech tag
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ,
                'N': wordnet.NOUN,
                'V': wordnet.VERB,
                'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Define a function to provide a recommendation based on user query
def recommend(query):
    # Preprocess the user query
    query = lemmatize(query.lower())
    keywords = ' '.join(query.split()[:3])
    topic = query.split()[3]
    difficulty = query.split()[-1]
    
    # Get the resource recommendation
    resource = rfc.predict([[keywords, topic, difficulty]])[0]
    
    # Convert the resource label back to text
    resource = le.inverse_transform([resource])[0]
    
    # Format the recommendation message
    message = 'I recommend the following resource for you: ' + resource
    
    return message

# Define a function to start the chatbot
def start_chatbot():
    print('Welcome to the Language Learning Chatbot!')
    print('Please enter your question or type "exit" to end the conversation.')
    while True:
        # Get user input
        user_input = input('You: ')
        # Exit the chatbot if user input is "exit"
        if user_input == 'exit':
            print('Thank you for using the Language Learning Chatbot!')
            break
        # Provide a recommendation based on user input
        else:
            response = recommend(user_input)
            print('Chatbot:', response)

start_chatbot()
