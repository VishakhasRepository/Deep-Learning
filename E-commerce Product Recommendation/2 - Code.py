# Step 1: Data Preprocessing

# Load the dataset
data = pd.read_csv('ecommerce_data.csv')

# Drop rows with missing values
data = data.dropna()

# Encode categorical variables
data = pd.get_dummies(data, columns=['category'])

# Scale numerical variables
scaler = StandardScaler()
data[['views', 'purchases']] = scaler.fit_transform(data[['views', 'purchases']])

# Step 2: Exploratory Data Analysis (EDA)

# Visualize the distribution of the target variable
sns.histplot(data, x='purchases')

# Visualize the relationship between features and the target variable
sns.scatterplot(data, x='views', y='purchases', hue='category')

# Step 3: Feature Engineering

# Perform dimensionality reduction
pca = PCA(n_components=2)
data[['pca1', 'pca2']] = pca.fit_transform(data[['views', 'purchases']])

# Create new features based on similarity to other products
similarity_matrix = cosine_similarity(data[['pca1', 'pca2']])
similar_products = np.argsort(-similarity_matrix, axis=1)[:, 1:4]
for i in range(3):
    data[f'similar_product_{i}'] = data.iloc[similar_products[:, i]]['product_id'].values

# Step 4: Model Selection

# Build collaborative filtering model
cf_model = NearestNeighbors(n_neighbors=3)
cf_model.fit(data[['views', 'purchases']])

# Build content-based filtering model
cb_model = RandomForestClassifier()
cb_model.fit(data.drop(['user_id', 'product_id', 'purchases'], axis=1), data['purchases'])

# Step 5: Model Tuning

# Tune hyperparameters of collaborative filtering model
params = {'n_neighbors': [3, 5, 7]}
cv = GridSearchCV(cf_model, params, scoring='neg_mean_squared_error')
cv.fit(data[['views', 'purchases']])
cf_model = cv.best_estimator_

# Step 6: Model Evaluation

# Evaluate the performance of the collaborative filtering model
metrics = cross_validate(cf_model, data[['views', 'purchases']], data['purchases'], cv=5, scoring=['precision', 'recall', 'f1'])
print(f"Precision: {np.mean(metrics['test_precision'])}")
print(f"Recall: {np.mean(metrics['test_recall'])}")
print(f"F1-Score: {np.mean(metrics['test_f1'])}")

