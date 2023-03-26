import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
hotel_data = pd.read_csv('hotel_data.csv')

# Remove irrelevant columns
hotel_data.drop(['hotel_name', 'address', 'city', 'state', 'postal_code'], axis=1, inplace=True)

# Normalize numerical features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(hotel_data.select_dtypes(include=[np.number]))

# Reduce dimensionality
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)

# Cluster hotels
kmeans = KMeans(n_clusters=5, random_state=0).fit(reduced_data)
hotel_data['cluster'] = kmeans.labels_

def get_recommendations(hotel_name):
    # Get cluster of input hotel
    input_cluster = hotel_data[hotel_data['hotel_name'] == hotel_name]['cluster'].values[0]
    
    # Filter hotels by cluster
    cluster_hotels = hotel_data[hotel_data['cluster'] == input_cluster]
    
    # Sort by rating and distance from input hotel
    cluster_hotels['distance'] = np.linalg.norm(cluster_hotels[['latitude', 'longitude']] - 
                                               hotel_data[hotel_data['hotel_name'] == hotel_name][['latitude', 'longitude']].values[0],
                                               axis=1)
    cluster_hotels.sort_values(by=['rating', 'distance'], ascending=False, inplace=True)
    
    # Return top 10 recommendations
    return cluster_hotels['hotel_name'].head(10).tolist()

hotel_name = 'Hotel Indigo Asheville Downtown'
recommendations = get_recommendations(hotel_name)
print(f'Recommended hotels for {hotel_name}:')
for recommendation in recommendations:
    print(recommendation)
