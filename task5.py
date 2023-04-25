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



### T is a list of points of the form [(x1,y1) ... (xn,yn)]

def trajectory_simplification(T, epsilon):
    # get the start and end points
    start_index = 0
    end_index = len(T) - 1
    dist_max = 0
    index_max = 0

    for i in range(start_index + 1, end_index):
        edge = (T[start_index],T[end_index])
        dist = distance(T[i],edge)

        if dist > dist_max:
            dist_max = dist
            index_max = i

    result = []
    if dist_max > epsilon:
        out_left = trajectory_simplification(T[:index_max+1], epsilon)
        result_left = []

        for point in out_left:
            if point not in result:
                result_left.append(point)
        result += result_left

        out_right = trajectory_simplification(T[index_max:], epsilon)
        result_right = []

        for point in out_right:
            if point not in result:
                result_right.append(point)
        result += result_right

    else:
        result += (T[0], T[-1])

    return result


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
    return 0


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


def random_partition(trajectory_set, k):
    # Create a randomized list of IDs based on the dictionary
    ids = list(trajectory_set.keys())
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

def closest_centroid(point, centroids):
    distances = [euc_distance(point, centroid) for centroid in centroids]
    return distances.index(min(distances))


#  trajectory_set --> a 2d array of trajectories
#  k --> number of clusters
#  max_iterations --> (t) max times the algorithm will run unless stopped before
#  seed_type --> the type of seeding algorithm

# TODO - we chose the kmeans++ algorithm. Implement it below:
def new_seeding(trajectory_dict, k):
    keys = list(trajectory_dict.keys())
    centroids = [trajectory_dict[random.choice(keys)]]

    for _ in range(k - 1):
        distances = []

        for key in keys:
            trajectory = trajectory_dict[key]

            min_distance = float('inf')
            for centroid in centroids:
                for point in trajectory:
                    distance = euc_distance(centroid, point)
                    if distance < min_distance:
                        min_distance = distance

            distances.append(min_distance)

        probabilities = [d ** 2 for d in distances]
        probabilities_sum = sum(probabilities)
        probabilities = [p / probabilities_sum for p in probabilities]

        selected_key = random.choices(keys, weights=probabilities, k=1)[0]
        centroids.append(trajectory_dict[selected_key])

    return centroids
"""
def lloyds_algorithm(trajectory_set, k, max_iterations, seed_type):
    # -------- 1. Partition T into k sets T1,...,Tk using the specified seed method.

    # k_set_ids is a dictionary where the keys are the group #, and values are array of trajectory IDs
    k_set_ids = {}

    if (seed_type == 'r'):
        k_set_ids = random_partition(trajectory_set, k)
    else:
        k_set_ids = new_seeding(trajectory_set, k)  # implement new seeding algo

    # -------- 2. Repeat either a) until t is reached, or b) partitions remain the same before and after an iteration
    for i in range(max_iterations):

        # ------------ STEP 0: Remake k groups, used for comparison later ---------------
        new_k_set = {}

        # ------------ STEP 1: Center computations ---------------

        # centers[0] stores the central trajectory of group 0 (with the actual points!!! not trajectory IDs)
        centers = {}

        # For each group #, compute the center
        for group in range(0, k - 1):

            k_group_trajectories = []  # 2d array of all trajectories in the group, each row is a trajectory

            for each_id in k_set_ids[group]:
                k_group_trajectories.append(trajectory_set[each_id])

            centers[group] = find_center_t2(k_group_trajectories)  # add the center as a trajectory

        # ------------ STEP 2: Reassignment ---------------
        # for each trajectory in the whole set, reassign it to whichever center trajectory is closest
        for traj_id in trajectory_set.keys():  # each row is a trajectory
            min_dist = Integer.MAX_VALUE
            min_center = 0  # the group in centers{} to choose
            for center in centers.keys():  # loop through keys 0 through k-1
                dist = distance(trajectory_set[traj_id], centers[center])  # input two trajectories with their points
                if dist < min_dist:
                    min_dist = dist
                    min_center = center
            # Now we have min_center! This is the group # (in range 0 to k-1) which this trajectory belongs to
            # Add traj to group # min_center
            if min_center not in new_k_set:
                new_k_set[min_center] = []
            new_k_set[min_center].append(traj_id)

        # ------------ STEP 3: Repeat, unless... : ---------------
        # Compare new_k_set and k_set_ids to see if anything changed
        difference = False
        for group in range(0, k - 1):
            new_list = new_k_set[group].sort()
            old_list = k_set_ids[group].sort()
            if len(new_list) == len(old_list):
                for i in range(0, len(new_list) - 1):  # is this the right range?
                    if new_list[i] != old_list[i]:
                        difference = True
            else:
                difference = True

        if difference == False:
            break
"""

#  trajectory_set --> a 2d array of trajectories
#  k --> number of clusters
#  max_iterations --> (t) max times the algorithm will run unless stopped before
#  seed_type --> the type of seeding algorithm

def lloyds_algorithm_v2(trajectory_set, k, max_iterations, seed_type):
    k_set_ids = {}

    if (seed_type == 'r'):
        k_set_ids = random_partition(trajectory_set, k)
    else:
        k_set_ids = new_seeding(trajectory_set, k)

    for i in range(max_iterations):
        new_k_set = {}
        centers = {}

        for group in range(0, k):
            k_group_trajectories = []
            for each_id in k_set_ids[group]:
                k_group_trajectories.append(trajectory_set[each_id])
            centers[group] = approach_2(k_group_trajectories)

        for traj_id in trajectory_set.keys():
            min_dist = float('inf')
            min_center = 0
            for center in centers.keys():
                dist = dtw(trajectory_set[traj_id], centers[center])
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

# Inputs are the same as Lloyd's algorithm
def get_cost(trajectory_set, k, max_iterations, seed_type):
    k_set_ids, centers = lloyds_algorithm_v2(trajectory_set, k, max_iterations, seed_type)

    cost = 0
    for group in k_set_ids.keys():  # keys are the group numbers, values are array of trajectory IDs
        for traj_id in group:
            cost = cost + distance(trajectory_set[traj_id], centers[group])
    return cost


if __name__ == '__main__':
    pass
    # print(data[:10])
    # print(lloyds(data, k, 100))
    d = make_dict("geolife-cars-upd8.csv")
    test2 = random_partition(d, 4)
    max_iterations = 100
    get_cost(test2, 4, max_iterations, 'r')