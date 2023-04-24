print("Hello world, task 5 here we go...")
### Implementing Random Seed for Lloyd's algo 
import numpy as np
import pandas as pd
import math
import random
import csv

"Raf and Yunner"
"Retest"


def make_dict(fname):
    T = {}
    counter = 0
    with open(fname, newline='', ) as f:
        reader = csv.reader(f)
        for row in reader:
            if counter != 0:
                if row[0] not in T:
                    T[row[0]] = []
                T[row[0]].append((float(row[1]), float(row[2])))
            counter += 1

    return T


def dtw(P, Q):
    lenP = len(P)
    lenQ = len(Q)
    # Initializing first dp table that will store the size of the assignment between P & Q
    memo = [[math.inf] * (lenQ + 1) for _ in range(lenP + 1)]
    memo[0][0] = 0
    # Initializing second dp table for averages
    dp_avg = [[math.inf] * (lenQ + 1) for _ in range(lenP + 1)]
    dp_avg[0][0] = 0
    # Creating the two tables that will be used to find path later
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
    # Reverse looping over dp_avg to add incdices to Eavg array (starting from bottom right corner)
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


def approach_2(points):
    # Compute the length of the longest trajectory
    # print(points)
    max_len = max([len(point) for point in points])
    # Initialize a matrix to store the points
    coordinates = [[[0, 0] for _ in range(len(points))] for _ in range(
        max_len)]
    # Fill the matrix with the points from each trajectory
    for i, point in enumerate(points):
        for j, coordinate in enumerate(point):
            coordinates[j][i] = coordinate
    # Compute the average point for each time step
    avg_points = [[sum(x) / len(x) for x in zip(*pt)] for pt in coordinates]
    # Return the center trajectory as a sequence of points
    # print(avg_points)
    return avg_points


def random_partition(dict_set, k):
    """need to change"""
    # Create a randomized list of IDs based on the dictionary
    ids = list(dict_set.keys())
    random.shuffle(ids)

    # Create a dictionary. Keys are groups in range(0, k-1). Values are array of trajectory IDs
    k_groups = {}
    group_size = len(ids) // k
    start = 0
    end = group_size
    for group_number in range(k):
        if group_number not in k_groups:
            k_groups[group_number] = []
        for i in range(start, end):
            k_groups[group_number].append(ids[i])
        start = end + 1
        if start + group_size > len(ids):
            end = len(ids) - 1
        else:
            end = start + group_size

    return k_groups


def assign(trajs, centers):
    print("made it to assign")
    clusters = dict.fromkeys(centers, [])
    # iterate through each traj P
    for id, P in trajs.items():
        min_dist, min_assign = math.inf, None
        print("first for loop")
        for center in centers:
            print("second for loop")
            Q = trajs[center]
            # compute dist to center Q
            dist = dtw(P, Q)[-1][-1]
            print("distance: ", dist)
            print("min_assign: ", min_assign)
            if dist < min_dist:
                min_dist = dist
                min_assign = center
                print("min assign: ", min_assign)
        # assign P to closest cluster Q
        clusters[min_assign].append(id)

    return clusters


def careful_seed(traj, k):
    return []


def update(trajs, clusters):
    new_center_ids = []
    for center, cluster in clusters.items():
        paths = [trajs[id] for id in cluster]
        new_center_path = approach_2(paths)
        trajs[len(new_center_ids)] = new_center_path
        new_center_ids.append(len(new_center_ids))

    return trajs, new_center_ids


def lloyds(trajs, k, t, seed):
    # initialize centers
    centers = random_seed(trajs, k) if seed == "random" else careful_seed(trajs, k)
    clusters = assign(trajs, centers)
    for iter in range(t):
        print("############## ITER: ", iter, " ##############")
        print("Finding new cluster centers...")
        trajs, centers = update(trajs, clusters)
        print("Updating cluster assignments...")
        new_clusters = assign(trajs, centers)
        if new_clusters == clusters:
            break
        else:
            clusters = new_clusters

    return clusters


def random_seed(trajs, k):
    return random.sample(trajs.keys(), k=k)


def get_trajectories(df):
    dict = {}
    for index, row in df.iterrows():
        # print(index / len(df))
        id, x, y = row[0], row[1], row[2]
        if id not in dict:
            dict[id] = [(float(x), float(y))]
        else:
            dict[id].append((float(x), float(y)))

    return dict


def get_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["date"])  # need with smaller datasets
    dict = get_trajectories(df)

    return dict



if __name__ == '__main__':
    pass
    # print(data[:10])
    # print(lloyds(data, k, 100))
    d = make_dict("geolife-cars-upd8.csv")
    test1 = random_seed(d, 4)
    test2 = random_partition(d, 4)
    print("test 1: ", test1, "\n, \n")
    print("test2 : ", test2)
    # print(d[:100], "\n \n")
    # print(get_trajectories())
    # print(lloyds(d, 4, 100, random))