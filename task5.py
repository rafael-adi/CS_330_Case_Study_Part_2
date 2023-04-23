print("Hello world, task 5 here we go...")

### Implementing Random Seed for Lloyd's algo 

import numpy as np
import math
import csv


"Raf and Yunner"


def dtw(P, Q):
    """Algorithm to find dtw and Eavg assignment; returns an array"""

    lenP = len(P)
    lenQ = len(Q)

    #Initializing first dp table that will store the size of the assignment between P & Q
    memo = [[math.inf] * (lenQ+1) for _ in range(lenP + 1)]
    memo[0][0] = 0

    #Initializing second dp table for averages
    dp_avg = [[math.inf] * (lenQ+1) for _ in range(lenP + 1)]
    dp_avg[0][0] = 0

    #Creating the two tables that will be used to find path later
    for i in range(1, lenP + 1):
        for j in range(1, lenQ + 1):
            distance = math.dist(P[i - 1], Q[j - 1])
            sqDistance = distance * distance

            temp1 = (sqDistance + (memo[i - 1][j - 1] * dp_avg[i - 1][j - 1])) / (memo[i - 1][j - 1] + 1)
            temp2 = (sqDistance + (memo[i][j - 1] * dp_avg[i][j - 1])) / (memo[i][j - 1] + 1)
            temp3 = (sqDistance + (memo[i - 1][j] * dp_avg[i - 1][j])) / (memo[i - 1][j] + 1)

            minimum = min(temp1, temp2, temp3)
            dp_avg[i][j] = minimum

    # Initializing Eavg array for indices
    Eavg = []

    i = lenP
    j = lenQ

    #Reverse looping over dp_avg to add incdices to Eavg array (starting from bottom right corner)
    while i > 0 and j > 0:
        temp1 = dp_avg[i - 1][j - 1]
        temp2 = dp_avg[i - 1][j]
        temp3 = dp_avg[i][j - 1]
        mintemp = min(temp1, temp2, temp3)
        if dp_avg[i - 1][j] == mintemp:
            Eavg.append([i - 1, j])
            i -= 1
        elif dp_avg[i - 1][j - 1] == mintemp:
            Eavg.append([i - 1, j - 1])
            i -= 1
            j -= 1
        else:
            Eavg.append([i, j - 1])
            j -= 1
    return Eavg[::-1]



def lloyd_kmeans_pp(data, k, max_iter=100):
    # Randomly choose first centroid from data points
    centroids = [data[np.random.choice(len(data))]]
    distances = np.zeros(len(data))
    
    # Choose remaining k-1 centroids using k-means++ seeding
    for _ in range(k - 1):
        # Compute distance between each data point and closest centroid
        for i in range(len(data)):
            dist, _ = dtw(data[i][1:], centroids[-1][1:])
            distances[i] = dist ** 2
            
        # Choose next centroid with probability proportional to square distance
        next_centroid_idx = np.random.choice(len(data), p=distances / np.sum(distances))
        centroids.append(data[next_centroid_idx])
        
    # Perform Lloyd's algorithm
    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        clusters = [[] for _ in range(k)]
        for i in range(len(data)):
            dists = [dtw(data[i][1:], c[1:])[0] for c in centroids]
            closest_centroid_idx = np.argmin(dists)
            clusters[closest_centroid_idx].append(data[i])
        
        # Recompute centroids as mean of points in each cluster
        for i in range(k):
            if clusters[i]:
                centroids[i] = [clusters[i][j][1:] for j in range(len(clusters[i]))]
                centroids[i] = np.mean(centroids[i], axis=0).tolist()
        
    return clusters, centroids


## Loading in Data Set 
data = []
with open('geolife-cars-upd8.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

k = 4

print(lloyd_kmeans_pp(data, k, 100))

