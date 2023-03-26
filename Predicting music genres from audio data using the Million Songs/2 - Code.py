# Import necessary libraries
import numpy as np
import pandas as pd
import os
import librosa
import sklearn

# Load the data
metadata = pd.read_csv('msd_genre_dataset.csv', usecols=['genre', 'track_id'])
tracks = pd.read_csv('tracks.csv', usecols=['track_id', 'genre_top'])

# Create a dictionary to map the track ids to their respective genres
track_genre_dict = {}
for index, row in metadata.iterrows():
    track_genre_dict[row['track_id']] = row['genre']
    
# Filter the tracks dataset to only include those with a defined genre
tracks = tracks[~tracks['genre_top'].isnull()]

# Create a new column in the tracks dataset that maps the track ids to their genres
tracks['genre'] = tracks['track_id'].map(track_genre_dict)

# Create a list of all the unique genre labels
genres = np.unique(tracks['genre'])

# Create a dictionary to map the genre labels to numerical values
label_dict = {}
for i, genre in enumerate(genres):
    label_dict[genre] = i
    
# Create a new column in the tracks dataset that maps the genre labels to numerical values
tracks['label'] = tracks['genre'].map(label_dict)

# Set the duration of the audio clips to extract features from
clip_duration = 30

# Initialize lists to store features and labels
features = []
labels = []

# Loop through each track in the tracks dataset
for index, row in tracks.iterrows():
    # Load the audio file using librosa
    filename = 'millionsongsubset_full/' + row['track_id'][2] + '/' + row['track_id'][3] + '/' + row['track_id'] + '.mp3'
    audio, sr = librosa.load(filename, duration=clip_duration)
    
    # Extract features using librosa
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(audio, sr=sr)
    mel = librosa.feature.melspectrogram(audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(audio, sr=sr)
    features.append(np.concatenate((mfccs.mean(axis=1), chroma.mean(axis=1), mel.mean(axis=1), contrast.mean(axis=1))))
    
    # Append the label to the labels list
    labels.append(row['label'])
    
# Convert the features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

# Fit the classifier to the training data
rfc.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = rfc.predict(X_test)

# Calculate the accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)

#Confusion matrix
test_predictions = model.predict(test_data)
test_predictions = np.argmax(test_predictions, axis=1)
conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#Classification report
print('Classification Report:\n')
print(classification_report(test_labels, test_predictions, target_names=genre_list))

#Plot ROC Curve
y_pred = model.predict(test_data)
fpr = {}
tpr = {}
thresh ={}
n_class = 10
for i in range(n_class):
fpr[i], tpr[i], thresh[i] = roc_curve(test_labels_onehot[:,i], y_pred[:,i])

plt.figure(figsize=(10,8))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
for i in range(n_class):
plt.plot(fpr[i], tpr[i], label=genre_list[i])
plt.legend()
plt.show()

#Save the model
model.save('music_genre_classifier.h5')