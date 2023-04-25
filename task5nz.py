import math 
import csv
import random 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# -- validated
def read_file(filename, ids=None):
    df = pd.read_csv(filename)

    df.sort_values(by=["id_", "date"], inplace=True)
    trajectories = {}   
    for _, row in df.iterrows():
        id = row["id_"]
        timestamp = row["date"]
        x = float(row["x"])
        y = float(row["y"])
        
        if id not in trajectories:
            trajectories[id] = []
        
        trajectories[id].append((x, y))

    return trajectories # returns a dict of trajectories (each trajectory is a list of tuples)


# -- validated: P and Q need to be tuples
def euclidean_distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

# -- validated by Rafa, not really validated
def random_seeding(trajectories, k):
    trajectory_ids = list(trajectories.keys())
    random.shuffle(trajectory_ids)

    ids = {}
    size = len(trajectory_ids) // k
    s = 0
    e = size

    for num in range(k):
        if num not in ids:
            ids[num] = []
        for i in range(s, e):
            ids[num].append(trajectory_ids[i])
        s = e + 1
        if len(trajectory_ids) < s + size:
            e = len(trajectory_ids) - 1
        else:
            e = s + size

    clusters = {}
    for id in trajectories.keys():
        if id not in clusters.keys():
            clusters[id] = trajectories[id]
        else:
            clusters[id].append(trajectories[id])

    print(len(clusters), clusters.keys())
    return clusters


# this works -- nate and yun
def random_seeding_v2(trajectories, k):
    trajectory_ids = list(trajectories.keys())

    clusters = {} # key: int from 0 to k-1, value: dictionary of trajectory ids, where each id maps to trajectory list of points
    for i in range(k): 
        clusters[i] = {} # init dict of dicts

    for id in trajectory_ids:
        cluster_id = random.randint(0, k-1) # assign trajectory to random cluster
        clusters[cluster_id][id] = trajectories[id]

    return clusters


# -- nate's dtw: O(n1 * n2)
def dtw(P, Q):
    n1 = len(P)
    n2 = len(Q)
    cost = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost[i,j] = euclidean_distance(P[i], Q[j])

    # Compute the accumulated cost matrix
    acc_cost = np.zeros((n1, n2))
    acc_cost[0,0] = cost[0,0]
    for i in range(1, n1):
        acc_cost[i,0] = acc_cost[i-1,0] + cost[i,0]
    for j in range(1, n2):
        acc_cost[0,j] = acc_cost[0,j-1] + cost[0,j]
    for i in range(1, n1):
        for j in range(1, n2):
            acc_cost[i,j] = cost[i,j] + min(acc_cost[i-1,j], acc_cost[i,j-1], acc_cost[i-1,j-1])

    # Find the optimal warping path
    path = []
    i = n1 - 1
    j = n2 - 1
    path.append((i,j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if acc_cost[i-1,j] == min(acc_cost[i-1,j-1], acc_cost[i-1,j], acc_cost[i,j-1]):
                i -= 1
            elif acc_cost[i,j-1] == min(acc_cost[i-1,j-1], acc_cost[i-1,j], acc_cost[i,j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i,j))
    path.reverse()
    dtw_distance = acc_cost[n1-1,n2-1]

    return dtw_distance, path


# -- NZ and WY
def center_computation(cluster, n_ticks=20):
    # print(cluster)
    # -- get x_min and x_max
    x_min, x_max = math.inf, -math.inf
    for id in cluster.keys():
        traj = cluster[id]
        for p in traj:
            # print(p, type(p), p[0], type(p[0]))
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]
    # print(x_min, x_max)

    # -- make linspace for x axis
    x_ticks = np.linspace(x_min, x_max, n_ticks)    
    y_values = np.zeros((len(cluster), n_ticks))
    for i, traj in enumerate(cluster):
        x_vals = np.array([p[0] for p in cluster[id]])
        y_vals = np.array([p[1] for p in cluster[id]])
        
        # Use linear interpolation to find y values at the x ticks
        interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate')
        y_values[i] = interp_func(x_ticks)
    
    # Average the y values across all trajectories at each x tick to construct the center trajectory
    center_y = np.mean(y_values, axis=0)
    
    # Construct the center trajectory as a list of (x, y) coordinate tuples
    center_traj = [(x_ticks[i], center_y[i]) for i in range(n_ticks)]
    return center_traj


def lloyds(TS, k, seed, max_iter=10):

    if seed == 'random':
        clusters = random_seeding_v2(TS, k) # clusters is a dict of dicts of trajectories, indexed 0 to k-1
    # elif seed == 'kmeans++':
    #     clusters = kmeans(TS, k) # clusters is a dict of dicts of trajectories, indexed 0 to k-1
    else:
        raise NotImplementedError
    
    clusters_old = clusters # keep record of old cluster to see if stopping condition is met
    centers = {} #key: indexed from 0 to k-1; val: center trajectory (list of tuples)
    counter = 0
    max_iter = 10 # CHANGE

    while counter < max_iter:
        # -- center computation
        for i in range(k):
            cluster = clusters[i]
            cluster_center = center_computation(cluster)  # clusters[i] is a dict of ids->trajectories
            centers[i] = cluster_center

        # -- reassignment
        for traj_id in TS.keys(): 
            traj = TS[traj_id]

            temp = {}
            for i in range(k):
                distance, _ = dtw(traj, centers[i])
                temp[i] = distance # populates dict: cluster_id -> distance
                if traj_id in clusters[i].keys(): # grab current cluster id
                    current_id = i
                    break
            
            # -- find min value in dict and grab cluster ID
            min_dist, min_id = math.inf, -1
            for key in temp.keys():
                if temp[key] < min_dist:
                    min_dist = temp[key]
                    min_id = key

            # -- assign trajectory to cluster with min_id
            if min_id != current_id:
                clusters[min_id][traj_id] = (traj)
                clusters[current_id][traj_id].pop()
    


        # -- check stopping condition
        if clusters == clusters_old:
            break
               # check if it is over max iterations
        counter += 1

    return clusters

def main():
    TS = read_file('geolife-cars-ten-percent.csv')  # CHANGE
    k = 3
    max_iter = 10

    result = lloyds(TS, k, 'random', max_iter)



if __name__ == '__main__':
    main()