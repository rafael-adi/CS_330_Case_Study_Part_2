import pandas as pd
import math as math
import os
import random
import matplotlib.pyplot as plt

### Dtw function 
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

### Distance Formula for Cost Function 
def euc_distance(p1,p2):
    return math.sqrt(((p1[0]-p2[0])**2) + ((p1[1]-p2[1])**2))

# Using this formula from: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
def distance(q,e):
    '''if q[0] < e[0][0]:
       return euc_distance(q,e[0])
    elif q[0] > e[1][0]:
       return euc_distance(q,e[1])
    else:'''
    top = abs(((e[1][0] - e[0][0]) * (e[0][1] - q[1])) - ((e[0][0] - q[0]) * (e[1][1] - e[0][1])))
    bottom = euc_distance(e[0],e[1])
    return top/bottom


def approach_2(points):
    # Compute the length of the longest trajectory
    #print(points)
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
    #print(avg_points)
    return avg_points

### data processing
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
    df = pd.read_csv(path)  # need with smaller datasets
    dict = get_trajectories(df)

    return dict


### Clustering
def random_seed(trajs, k):
    return random.sample(trajs.keys(), k=k)

### Need proposed seeding method 
def careful_seed(traj, k):
    return []


def assign(trajs, centers):
    clusters = dict.fromkeys(centers, [])
    # iterate through each traj P
    for id, P in trajs.items():
        min_dist, min_assign = math.inf, None
        for center in centers:
            Q = trajs[center]
            # compute dist to center Q
            dist = dtw(P, Q)[-1][-1]
            if dist < min_dist:
                min_dist = dist
                min_assign = center
        # assign P to closest cluster Q
        clusters[min_assign].append(id)

    return clusters


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

#Inputs are the same as Lloyd's algorithm
### Need some sort of cost function 


#### From Task 2 to speed up runtime 

def d(q, e):
    """
      Compute the distance between a point q and a segment e
    """
    a = e[0]
    b = e[1]
    ax = a[0]
    ay = a[1]
    bx = b[0]
    by = b[1]
    qx = q[0]
    qy = q[1]

    # vector ab
    ab_x = bx - ax
    ab_y = by - ay

    # vector aq
    aq_x = qx - ax
    aq_y = qy - ay

    # compute dot product of ab and aq
    dp = (ab_x * aq_x) + (ab_y * aq_y)
    abLen = abs(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
    q_to_AB = dp / (abLen ** 2)

    if (q_to_AB >= 1):  # closest point is B
        return abs(math.sqrt((qx - bx) ** 2 + (qy - by) ** 2))

    elif (q_to_AB <= 0):  # closest point is A
        return abs(math.sqrt((qx - ax) ** 2 + (qy - ay) ** 2))

    else:  # closest point is in the middle of line segment e, let f be that point
        fx = (ab_x * q_to_AB) + ax
        fy = (ab_y * q_to_AB) + ay
        return abs(math.sqrt((qx - fx) ** 2 + (qy - fy) ** 2))


def greedy(T, ep):
    """
      Implement the greedy algorithm TS-greedy(T,ε) to compute an ε-simplification of T.

      Algo: essentially iterates through each point in the trajectory, checking if
    """

    traj = [T[0], T[-1]]

    for i in range(1, len(T) - 1):
        dist = d(T[i], [T[0], T[-1]])

        if dist > ep:
            left_traj = greedy(T[:i + 1], ep)
            right_traj = greedy(T[i:], ep)

            count = len(traj)
            traj = left_traj[:-1] + right_traj

            break

    return traj

### To speed up runtime 
def simplify_trajectories(dict):
    e = 0.3  # maximum error value for trajectory simplification
    for key in dict.keys():
        ts = greedy(dict[key], e)
        dict[key] = ts
    return dict

#Cost of clustering k = 4, Proposed seeding



costs_random = {}

if __name__ == "__main__":
    # hyperparams
    file = "geolife-cars-upd8.csv"  # "geolife-cars-upd8.csv"
    path = os.path.join(file)
    k = 4
    t = 100
    # read in trajectories
    trajs = get_data(path)
    # simplify trajectories
    simp_trajs = simplify_trajectories(trajs)
    # run lloyds
    clusters = lloyds(simp_trajs, k, t, seed="random")
    print(clusters)
    ### Need plot 

    


