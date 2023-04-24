print("Hello world, task 5 here we go...")
### Implementing Random Seed for Lloyd's algo 
import numpy as np
import math
import random
import csv

"Raf and Yunner"


def make_dict(fname):
   T = {}
   counter = 0
   with open(fname, newline='',) as f:
       reader = csv.reader(f)
       for row in reader:
           if counter != 0:
               if row[0] not in T:
                   T[row[0]] = []
               T[row[0]].append((row[1], row[2]))
           counter += 1

   return T

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
def lloyds(data, k, tmax=100):
    # Initialize centroids with k-means++
    centroids = [data[0]]
    print(centroids)
    for i in range(1, k):
        distances = np.zeros(len(data))
        for j in range(len(data)):
            print(centroids[-1], data[j])
            dist, _ = dtw(centroids[-1], data[j])
            distances[j] = dist
        probabilities = distances / distances.sum()
        cum_probabilities = probabilities.cumsum()
        r = np.random.rand()
        print("checkpoint 1")
        for j, p in enumerate(cum_probabilities):
            if r < p:
                centroids.append(data[j])
                break
        print("checkpoint 2")
    # Assign points to nearest centroid
    assignments = np.zeros(len(data))
    t = 0
    while t < tmax:
        t += 1
        for i in range(len(data)):
            distances = np.zeros(k)
            for j in range(k):
                dist, _ = dtw(data[i], centroids[j])
                distances[j] = dist
            assignments[i] = np.argmin(distances)
        print("checkpoint 3")
        # Update centroids
        for j in range(k):
            centroid_points = data[assignments == j]
            if len(centroid_points) > 0:
                centroids[j] = centroid_points.mean(axis=0)
        print("checkpoint 4")
    return assignments, centroids
## Loading in Data Set 
## Loading in Data Set
data = []
with open('geolife-cars-upd8.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the first row
    for row in reader:
        id = row[0]
        x = float(row[1])
        y = float(row[2])
        data.append([id, x, y])
k = 4
print(data[:10])
print(lloyds(data, k, 100))



if __name__ == '__main__':
    pass
    print(make_dict("geolife-cars-upd8.csv"))