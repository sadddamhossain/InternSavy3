#!/usr/bin/env python
# coding: utf-8

# # import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Mall_Customers.csv")


# In[3]:


df


# # Analyze the data

# In[22]:


df.info()


# In[11]:


df.isnull()


# In[21]:


df.isnull().sum()


# In[10]:


print(df.columns)


# In[12]:


df.describe()


# In[13]:


print(df.dtypes)


# In[15]:


duplicate_rows = df[df.duplicated()]
print(duplicate_rows)


# In[20]:


# Create a male DataFrame while excluding the 'cluster' column
male_df = df[df['Gender'] == 'Male']
# Print the separate male dataset
print("Male Dataset:")
print(male_df)


# In[17]:


female_df = df[df['Gender'] == 'Female']

print("\nFemale Dataset:")
print(female_df)


# # Customer Segmentation Model

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[24]:


# Select relevant features for segmentation
X = df[['Age', 'Annual Income (k$)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[25]:


# Determine the optimal number of clusters using the Elbow method
wcss = []

# Try different numbers of clusters from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# In[26]:


# Plot the Elbow method to find the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()


# In[27]:


# Based on the Elbow plot, choose the optimal number of clusters (e.g., 4)
n_clusters = 4

# Perform K-means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)


# In[32]:


# Add the cluster labels to your original DataFrame
df['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], label=f'Cluster {i+1}')

    # Annotate some points in each cluster (you can adjust this as needed)
    if not cluster_data.empty:
        sample_points = cluster_data.sample(min(5, len(cluster_data)))  # Sample up to 5 points
        for index, row in sample_points.iterrows():
            plt.annotate(f'ID: {row["CustomerID"]}', (row['Age'], row['Annual Income (k$)']), fontsize=8, alpha=0.7)

plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()


# In[29]:


# Plot cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()


# In[30]:


# You can analyze the clusters further to understand customer characteristics
for i in range(n_clusters):
    cluster_data = df[df['Cluster'] == i]
    print(f"Cluster {i+1}:")
    print(cluster_data.describe())


# In[ ]:




