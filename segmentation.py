import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Step 1: Load data
try:
    data = pd.read_csv('CC_GENERAL.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: File 'CC_GENERAL.csv' not found.")
    exit()

# Step 2: Preprocess data
features = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
X = data[features].fillna(0)  # Fill missing values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize clusters
fig = px.scatter(data, x='BALANCE', y='PURCHASES', color='Cluster',
                 title='Customer Segments by Balance and Purchases',
                 labels={'Cluster': 'Segment'})
fig.write_image('segments.png')
fig.show()

# Step 5: Save cluster stats
cluster_stats = data.groupby('Cluster')[features].mean().round(2)
cluster_stats.to_csv('cluster_stats.csv')
print("Cluster Stats:\n", cluster_stats)