print("Hello world, task 5 here we go...")
### Implementing Random Seed for Lloyd's algo 
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import itertools

"Raf and Yunner"
"Retest"

def euc_dist(P,Q):
    return math.sqrt(((P[0]-Q[0])**2) + ((P[1]-Q[1])**2))

def dist(P,Q):
    numer = abs(((Q[1][0] - Q[0][0]) * (Q[0][1] - P[1])) - ((Q[0][0] - P[0]) * (Q[1][1] - Q[0][1])))
    euclidian = euc_dist(Q[0],Q[1])
    return numer/euclidian


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

    memo = [[math.inf] * (lenQ+1) for _ in range(lenP + 1)]
    memo[0][0] = 0
    for i in range(1, lenP + 1):
        for j in range(1, lenQ + 1):
            euc = euc_dist(P[i - 1], Q[j - 1])
            temp1 = memo[i - 1][j]
            temp2 = memo[i - 1][j - 1]
            temp3 =  memo[i][j - 1]
            memo[i][j] = euc + min(temp1, temp2, temp3)

    return memo[lenP][lenQ]

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


def random_seeding(trajectories, k):
    trajectory_ids = list(trajectories.keys())
    random.shuffle(trajectory_ids)

    groups = {}
    size = len(trajectory_ids) // k
    s = 0
    e = size

    for num in range(k):
        if num not in groups:
            groups[num] = []
        for i in range(s, e):
            groups[num].append(trajectory_ids[i])
        s = e + 1
        if len(trajectory_ids) < s + size:
            e = len(trajectory_ids) - 1
        else:
            e = s + size

    return groups

"""
def closest_centroid(point, centroids):
    distances = [euc_dist(point, centroid) for centroid in centroids]
    return distances.index(min(distances))
"""

def proposed_seeding(trajectories, k):
    trajectory_ids = list(trajectories.keys())
    centroids = [trajectories[random.choice(trajectory_ids)]]

    for _ in range(k - 1):
        dists = []

        for id in trajectory_ids:
            trajectory = trajectories[id]

            min_distance = float('inf')
            for c in centroids:
                for p in trajectory:
                    euc_distance = euc_dist(c, p)
                    if min_distance > euc_distance:
                        min_distance = euc_distance

            dists.append(min_distance)

        probs = [d ** 2 for d in dists]
        probs_total = sum(probs)
        new_probs = [p / probs_total for p in probs]

        chosen_id = random.choices(trajectory_ids, weights=new_probs, k=1)[0]
        centroids.append(trajectories[chosen_id])

    return centroids

def lloyds_algorithm(trajectories, k, max_iterations, seed):
    #  trajectories is a dictionary w key as id and value as points
    #  k is the number of clusters
    #  max_iterations is the maximum times the algorithm will run
    #  seed is the type of seeding algorithm (random or proposed)

    if (seed == 'r'):
        k_ids = random_seeding(trajectories, k)
    else:
        k_ids = proposed_seeding(trajectories, k)

    for i in range(max_iterations):
        new_set = {}
        cluster_centers = {}

        for group in range(0, k):
            trajectory_groups = []
            for id in k_ids[group]:
                trajectory_groups.append(trajectories[id])
            cluster_centers[group] = approach_2(trajectory_groups)

        for id in trajectories.keys():
            final_center = 0
            final_dist = float('inf')

            for c_c in cluster_centers.keys():
                temp_dist = dtw(trajectories[id], cluster_centers[c_c])
                if temp_dist < final_dist:
                    final_dist = temp_dist
                    final_center = c_c

            if final_center not in new_set:
                new_set[final_center] = []

            new_set[final_center].append(id)

        boolean = False
        for g in range(0, k):
            new_list = sorted(new_set[g])
            old_list = sorted(k_ids[g])
            if len(old_list) == len(new_list):
                for i in range(len(old_list)):
                    if old_list[i] != new_list[i]:
                        boolean = True
            else:
                boolean = True

        if boolean == False:
            break

    return k_ids, cluster_centers


def find_cost(trajectories, k, max_iterations, seed):
    k_ids, centers = lloyds_algorithm(trajectories, k, max_iterations, seed)

    cost = 0
    for id_group in k_ids.keys():
        for trajectory in id_group:
            cost = cost + dist(trajectories[trajectory], centers[id_group])
    return cost

def finding_random_seed_costs():
    max_iterations = 100
    random_seed_costs = {}
    trajectories = make_dict("geolife-cars-upd8.csv")

    costs_4 = []
    for i in range(3):
        costs_4.append(find_cost(trajectories, 4, max_iterations, 'r'))

    numer_4 = costs_4[0] + costs_4[1] + costs_4[2]
    random_seed_costs[4] = numer_4 / 3

    costs_6 = []
    for i in range(3):
        costs_6.append(find_cost(trajectories, 6, max_iterations, 'r'))

    numer_6 = costs_6[0] + costs_6[1] + costs_6[2]
    random_seed_costs[6] = numer_6 / 3

    costs_8 = []
    for i in range(3):
        costs_8.append(find_cost(trajectories, 8, max_iterations, 'r'))

    numer_8 = costs_8[0] + costs_8[1] + costs_8[2]
    random_seed_costs[8] = numer_8 / 3

    costs_10 = []
    for i in range(3):
        costs_10.append(find_cost(trajectories, 10, max_iterations, 'r'))

    numer_10 = costs_10[0] + costs_10[1] + costs_10[2]
    random_seed_costs[10] = numer_10 / 3

    costs_12 = []
    for i in range(3):
        costs_12.append(find_cost(trajectories, 12, max_iterations, 'r'))

    numer_12 = costs_12[0] + costs_12[1] + costs_12[2]
    random_seed_costs[12] = numer_12 / 3

    return random_seed_costs

def finding_proposed_seed_costs():
    max_iterations = 20
    proposed_seed_costs = {}
    trajectories = make_dict("geolife-cars-upd8.csv")

    costs_4 = []
    for i in range(3):
        costs_4.append(find_cost(trajectories, 4, max_iterations, 'p'))

    numer_4 = costs_4[0] + costs_4[1] + costs_4[2]
    proposed_seed_costs[4] = numer_4 / 3

    costs_6 = []
    for i in range(3):
        costs_6.append(find_cost(trajectories, 6, max_iterations, 'p'))

    numer_6 = costs_6[0] + costs_6[1] + costs_6[2]
    proposed_seed_costs[6] = numer_6 / 3

    costs_8 = []
    for i in range(3):
        costs_8.append(find_cost(trajectories, 8, max_iterations, 'p'))

    numer_8 = costs_8[0] + costs_8[1] + costs_8[2]
    proposed_seed_costs[8] = numer_8 / 3

    costs_10 = []
    for i in range(3):
        costs_10.append(find_cost(trajectories, 10, max_iterations, 'p'))

    numer_10 = costs_10[0] + costs_10[1] + costs_10[2]
    proposed_seed_costs[10] = numer_10 / 3

    costs_12 = []
    for i in range(3):
        costs_12.append(find_cost(trajectories, 12, max_iterations, 'p'))

    numer_12 = costs_12[0] + costs_12[1] + costs_12[2]
    proposed_seed_costs[12] = numer_12 / 3

    return proposed_seed_costs

def plot_random(costs_random):
    x = list(costs_random.keys())
    y = list(costs_random.values())
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel('AVG Cost at K')
    plt.title('Cost of Lloyds with Random Seeding')
    plt.show()

def plot_proposed(costs_proposed):
    x = list(costs_proposed.keys())
    y = list(costs_proposed.values())
    plt.plot(x, y)
    plt.xlabel('K Value')
    plt.ylabel('Average Cost at K')
    plt.title('Cost of Lloyds with Proposed Seeding')
    plt.show()


if __name__ == '__main__':
    trajectories = make_dict("geolife-cars-upd8.csv")
    k = 4
    max_iterations = 5
    seed = 'r'
    print(lloyds_algorithm(trajectories, k, max_iterations, seed))
    #print(plot_random(finding_random_seed_costs()))