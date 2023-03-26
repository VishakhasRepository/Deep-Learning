# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Section 1: Data Preprocessing

# Load the raw data from the CSV files
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# Remove missing or invalid data points
def remove_invalid_data(df):
    df = df.dropna()
    return df

# Normalize the data using standardization
def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Section 2: Baseline Anomaly Detection Algorithms

# Implement a mean baseline anomaly detection algorithm
def mean_baseline(X, window_size):
    ma = X.rolling(window=window_size).mean()
    mstd = X.rolling(window=window_size).std()
    upper_bound = ma + 2*mstd
    lower_bound = ma - 2*mstd
    anomalies = []
    for i in range(len(X)):
        if X[i] > upper_bound[i] or X[i] < lower_bound[i]:
            anomalies.append(i)
    return anomalies

# Implement a change point detection baseline anomaly detection algorithm
def change_point_detection(X):
    from ruptures import rpt
    model = "l2"  # "l1", "rbf"
    algo = rpt.Dynp(model=model, min_size=1, jump=5).fit(X)
    result = algo.predict(pen=10)
    anomalies = result[:-1]
    return anomalies

# Section 3: Advanced Anomaly Detection Algorithms

# Implement an autoencoder anomaly detection algorithm
def autoencoder(X, encoding_dim=20, num_epochs=50, batch_size=16):
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers

    input_dim = X.shape[1]

    # Define the encoder layer
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)

    # Define the decoder layer
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Define the autoencoder model
    autoencoder_model = Model(inputs=input_layer, outputs=decoded)

    # Define the encoder model
    encoder_model = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder model
    autoencoder_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder model
    autoencoder_model.fit(X, X, epochs=num_epochs, batch_size=batch_size, shuffle=True)

    # Use the encoder model to get the encoded data
    encoded_data = encoder_model.predict(X)

    # Calculate the reconstruction error
    mse = np.mean(np.power(X - encoded_data, 2), axis=1)
    threshold = np.mean(mse) + np.std(mse)*2.5

    # Identify the anomalies
    anomalies = np.where(mse > threshold)[0]

    return anomalies

# Section 4: Model Evaluation

# Calculate the precision-recall curve and AUC
def calculate_pr_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    return pr_auc

# Calculate the ROC curve and AUC
#Calculate the ROC curve and AUC
def calculate_roc_curve(y_true, y_pred):
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
return roc_auc

#Plot the ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title):
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(title)
plt.legend(loc="lower right")
plt.show()

#Section 5: Results and Visualization
#Load the NAB dataset
data = load_data('data/realTweets/Twitter_volume_AMZN.csv')

#Remove any missing or invalid data points
data = remove_invalid_data(data)

#Normalize the data using standardization
X = data['value'].values.reshape(-1, 1)
X_scaled = normalize_data(X)

#Run the mean baseline anomaly detection algorithm
anomalies_mean = mean_baseline(X, window_size=100)

#Run the change point detection baseline anomaly detection algorithm
anomalies_cp = change_point_detection(X)

#Run the autoencoder anomaly detection algorithm
anomalies_ae = autoencoder(X_scaled)

#Evaluate the performance of the algorithms using the NAB evaluation metrics
y_true = np.zeros(X.shape[0])
y_true[NAB_DATA['anomaly'].values == 1] = 1

y_pred_mean = np.zeros(X.shape[0])
y_pred_mean[anomalies_mean] = 1

y_pred_cp = np.zeros(X.shape[0])
y_pred_cp[anomalies_cp] = 1

y_pred_ae = np.zeros(X.shape[0])
y_pred_ae[anomalies_ae] = 1

pr_auc_mean = calculate_pr_curve(y_true, y_pred_mean)
pr_auc_cp = calculate_pr_curve(y_true, y_pred_cp)
pr_auc_ae = calculate_pr_curve(y_true, y_pred_ae)

fpr_mean, tpr_mean, _ = roc_curve(y_true, y_pred_mean)
roc_auc_mean = calculate_roc_curve(y_true, y_pred_mean)

fpr_cp, tpr_cp, _ = roc_curve(y_true, y_pred_cp)
roc_auc_cp = calculate_roc_curve(y_true, y_pred_cp)

fpr_ae, tpr_ae, _ = roc_curve(y_true, y_pred_ae)
roc_auc_ae = calculate_roc_curve(y_true, y_pred_ae)

#Visualize the results of the anomaly detection algorithms using plots and graphs
plot_roc_curve(fpr_mean, tpr_mean, roc_auc_mean, 'ROC Curve (Mean Baseline)')
plot_roc_curve(fpr_cp, tpr_cp, roc_auc_cp, 'ROC Curve (Change Point Detection)')
plot_roc_curve(fpr_ae, tpr_ae, roc_auc_ae, 'ROC Curve (Autoencoder)')
#
#Print the performance metrics
print('Mean Baseline - Precision-Recall AUC:', pr_auc_mean)
print('Change Point Detection - Precision-Recall AUC:', pr_auc_cp)
print('Autoencoder - Precision-Recall AUC:', pr_auc_ae)
print('Mean Baseline - ROC AUC
