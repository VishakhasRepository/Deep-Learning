import pandas as pd
from sklearn.model_selection import train_test_split

# Load the METR-LA traffic dataset
df = pd.read_csv('METR-LA/metr-la.h5', sep='\t')

# Drop the 'Unnamed: 0' column
df = df.drop('Unnamed: 0', axis=1)

# Remove missing or invalid data points
df = df.dropna()

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, shuffle=False)

import numpy as np

# Engineer the features for the deep learning model
def create_features(df):
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Add time-based features
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday

    # Add sensor-specific features
    sensor_ids = np.unique(df['sensor_id'])
    for sensor_id in sensor_ids:
        df_sensor = df[df['sensor_id'] == sensor_id]
        df_sensor['rolling_mean'] = df_sensor['flow'].rolling(5).mean()
        df_sensor['rolling_std'] = df_sensor['flow'].rolling(5).std()
        df_sensor['diff'] = df_sensor['flow'].diff()

        df[df['sensor_id'] == sensor_id] = df_sensor

    return df

# Create the features for the training and testing sets
train = create_features(train)
test = create_features(test)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# Define the LSTM model for traffic flow prediction
def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Define the hyperparameters to tune
batch_size = [32, 64, 128]
epochs = [10, 50, 100]

# Perform grid search for hyperparameter tuning
for bs in batch_size:
    for ep in epochs:
        model = lstm_model(input_shape=(1, 5))
        model.fit(train[['hour', 'weekday', 'rolling_mean', 'rolling_std', 'diff']], train['flow'],
                  batch_size=bs, epochs=ep, callbacks=[EarlyStopping(patience=10)], verbose=0)
        score = model.evaluate(test[['hour', 'weekday', 'rolling_mean', 'rolling_std', 'diff']], test['flow'], verbose=0)
        print(f'Batch size: {bs}, Epochs: {ep}, Test Score: {score}')

# Train the LSTM model on the training data
model = lstm_model(input_shape=(1, 5))
model.fit(train[['hour', 'weekday', 'rolling_mean', 'rolling_std', 'diff']], train['flow'],
          batch_size=32, epochs=50, callbacks=[EarlyStopping(patience=10)], verbose=0)

# Evaluate the model on the testing data
test_predictions = model.predict(test[['hour', 'weekday', 'rolling_mean',

# Evaluate the model on the testing data
test_predictions = model.predict(test[['hour', 'weekday', 'rolling_mean', 'rolling_std', 'diff']])
test_predictions = test_predictions.reshape(-1)

# Calculate the mean squared error
mse = mean_squared_error(test['flow'], test_predictions)
print('Test MSE: ', mse)

# Calculate the mean absolute error
mae = mean_absolute_error(test['flow'], test_predictions)
print('Test MAE: ', mae)

# Calculate the R-squared value
r2 = r2_score(test['flow'], test_predictions)
print('R2 Score: ', r2)


