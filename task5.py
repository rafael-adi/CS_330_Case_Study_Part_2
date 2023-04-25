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
    return


def approach_2(points):
    return


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
    k_set_ids = {}

    if (seed == "r"):
        k_set_ids = random_seeding(trajectories, k)
    elif (seed == "p"):
        k_set_ids = proposed_seeding(trajectories, k)

    for i in range(max_iterations):
        new_k_set = {}
        centers = {}

        for group in range(0, k):
            k_group_trajectories = []
            for each_id in k_set_ids[group]:
                k_group_trajectories.append(trajectories[each_id])
            centers[group] = approach_2(k_group_trajectories)

        for traj_id in trajectories.keys():
            min_dist = float('inf')
            min_center = 0
            for center in centers.keys():
                dist = dtw(trajectories[traj_id], centers[center])
                if dist < min_dist:
                    min_dist = dist
                    min_center = center
            if min_center not in new_k_set:
                new_k_set[min_center] = []
            new_k_set[min_center].append(traj_id)

        difference = False
        for group in range(0, k):
            new_list = sorted(new_k_set[group])
            old_list = sorted(k_set_ids[group])
            if len(new_list) == len(old_list):
                for i in range(len(new_list)):
                    if new_list[i] != old_list[i]:
                        difference = True
            else:
                difference = True

        if difference == False:
            break

    return k_set_ids, centers

# Inputs are the same as Lloyd's algorithm
def get_cost(trajectory_set, k, max_iterations, seed_type):
    k_set_ids, centers = lloyds_algorithm(trajectory_set, k, max_iterations, seed_type)

    cost = 0
    for group in k_set_ids.keys():  # keys are the group numbers, values are array of trajectory IDs
        for traj_id in group:
            cost = cost + dist(trajectory_set[traj_id], centers[group])
    return cost

def finding_costs_random():
    max_iterations = 100
    costs_random = {}
    trajectories = make_dict("geolife-cars-upd8.csv")

    costs_random_4 = []
    for i in range(3):
        costs_random_4.append(get_cost(trajectories, 4, max_iterations, 'r'))

    costs_random[4]/(costs_random_4[0] + costs_random_4[1] + costs_random_4[2]) / 3

    costs_random_6 = []
    for i in range(3):
        costs_random_6.append(get_cost(trajectories, 6, max_iterations, 'r'))

    costs_random[6] / (costs_random_4[0] + costs_random_4[1] + costs_random_4[2]) / 3

    costs_random_8 = []
    for i in range(3):
        costs_random_8.append(get_cost(trajectories, 8, max_iterations, 'r'))

    costs_random[8] / (costs_random_4[0] + costs_random_4[1] + costs_random_4[2]) / 3

    costs_random_10 = []
    for i in range(3):
        costs_random_10.append(get_cost(trajectories, 10, max_iterations, 'r'))

    costs_random[10] / (costs_random_4[0] + costs_random_4[1] + costs_random_4[2]) / 3

    costs_random_12 = []
    for i in range(3):
        costs_random_12.append(get_cost(trajectories, 12, max_iterations, 'r'))

    costs_random[12] / (costs_random_4[0] + costs_random_4[1] + costs_random_4[2]) / 3

    return costs_random

def finding_costs_proposed():
    max_iterations = 100
    costs_proposed = {}
    trajectories = make_dict("geolife-cars-upd8.csv")

    costs_proposed_4 = []
    for i in range(3):
        costs_proposed_4.append(get_cost(trajectories, 4, max_iterations, 'r'))

    costs_proposed[4] / (costs_proposed_4[0] + costs_proposed_4[1] + costs_proposed_4[2]) / 3

    costs_proposed_6 = []
    for i in range(3):
        costs_proposed_6.append(get_cost(trajectories, 6, max_iterations, 'r'))

    costs_proposed[6] / (costs_proposed_6[0] + costs_proposed_6[1] + costs_proposed_6[2]) / 3

    costs_proposed_8 = []
    for i in range(3):
        costs_proposed_8.append(get_cost(trajectories, 8, max_iterations, 'r'))

    costs_proposed[8] / (costs_proposed_8[0] + costs_proposed_8[1] + costs_proposed_8[2]) / 3

    costs_proposed_10 = []
    for i in range(3):
        costs_proposed_10.append(get_cost(trajectories, 10, max_iterations, 'r'))

    costs_proposed[10] / (costs_proposed_10[0] + costs_proposed_10[1] + costs_proposed_10[2]) / 3

    costs_proposed_12 = []
    for i in range(3):
        costs_proposed_12.append(get_cost(trajectories, 12, max_iterations, 'r'))

    costs_proposed[12] / (costs_proposed_12[0] + costs_proposed_12[1] + costs_proposed_12[2]) / 3

    return costs_proposed

def plot_random(costs_random):
    random_x = list(costs_random.keys())
    random_y = list(costs_random.values())
    plt.plot(random_x, random_y)
    plt.xlabel('K Value')
    plt.ylabel('Average Cost at K')
    plt.title('Cost of Lloyds Algorithm with Random Seeding at Different Values of K')
    plt.show()

def plot_proposed(costs_proposed):
    random_x = list(costs_proposed.keys())
    random_y = list(costs_proposed.values())
    plt.plot(random_x, random_y)
    plt.xlabel('K Value')
    plt.ylabel('Average Cost at K')
    plt.title('Cost of Lloyds Algorithm with Updated Seeding at Different Values of K')
    plt.show()


if __name__ == '__main__':
    d = make_dict("geolife-cars-upd8.csv")
    test2 = random_seeding(d, 4)
    max_iterations = 100
    get_cost(test2, 4, max_iterations, 'r')