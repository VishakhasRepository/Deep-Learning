# Importing the Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# Loading the Dataset
data = pd.read_csv('ieee-fraud-detection/train_transaction.csv')

# Data Cleaning and Preprocessing
data.drop(['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD'], axis=1, inplace=True)
data.fillna(-999, inplace=True)
data = pd.get_dummies(data, columns=['card4', 'card6', 'P_emaildomain', 'R_emaildomain'])

# Splitting the Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(data.drop('isFraud', axis=1), data['isFraud'], test_size=0.2, random_state=0)

# Model Creation
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.5, colsample_bytree=0.5, gamma=1, random_state=0)
xgb.fit(X_train, y_train)

# Model Evaluation
y_pred_prob = xgb.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC = {:.2f})'.format(auc_score))
plt.show()

y_pred = xgb.predict(X_val)
conf_mat = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

# Predictions on Test Data
test_data = pd.read_csv('ieee-fraud-detection/test_transaction.csv')
test_data.drop(['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD'], axis=1, inplace=True)
test_data.fillna(-999, inplace=True)
test_data = pd.get_dummies(test_data, columns=['card4', 'card6', 'P_emaildomain', 'R_emaildomain'])
y_test_pred = xgb.predict_proba(test_data)[:, 1]
submission = pd.DataFrame({'TransactionID': pd.read_csv('ieee-fraud-detection/test_transaction.csv')['TransactionID'], 'isFra
