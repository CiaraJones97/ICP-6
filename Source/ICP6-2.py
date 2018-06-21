import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans # Algorithm for the clustering

df = pd.read_csv ('sample_stocks.csv', header=0, delimiter=",") # reading csv file

arr = np.array(df) # Converting the input data into numpy array
k = len (arr)//2 # To determine a good estimate for k (number of clusters)
k = k** 0.5
kmeans = KMeans (n_clusters= int(k)) # Number of clusters to be used
kmeans.fit(arr) # Fitting the data into the model
centroids = kmeans.cluster_centers_ # Calculating the centroids

print(k)
plt.scatter(arr[:,0],arr[:,1]) # Plotting the input data
for i in range(int(k)):
    plt.scatter(centroids[i][0],centroids[i][1], marker = "x", linewidths=20,) # ploting the centroid
plt.show()

"""
The elbo method can be used to choose the best(more precise) k value (number of clusters):
    - compute the sum of squared error for some values of k
    - The sum of squared distance between each member of the cluster and its centroid
    - as k increases the value of sum of squared error decreases
    - at certain increament of k the change in sum of squared result will be insignificant
    - that point will be the best choice of k 
"""
