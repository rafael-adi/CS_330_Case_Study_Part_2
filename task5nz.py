import math 
import csv
import random 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def read_file(filename, ids=None):
    df = pd.read_csv(filename)

    df.sort_values(by=["id_"], inplace=True)
    trajectories = {}   
    for _, row in df.iterrows():
        id = row["id_"]
        x = float(row["x"])
        y = float(row["y"])
        
        if id not in trajectories:
            trajectories[id] = []
        
        trajectories[id].append((x, y))

    return trajectories # returns a dict of trajectories (each trajectory is a list of tuples)



def euclidean_distance(p, q):
    #euclidian distance function
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

### propose_seed uses Kmeans++ seeding to create clusters that will go into the Lloyd's algorithm method
def proposed_seed(trajectories, k):
    trajectory_ids = list(trajectories.keys()) #set up id's as list
    trajectories_list = list(trajectories.values()) #set up all values as list
    centers = []
    centers.append(random.choice(trajectories_list)) #create random starting centroid

    for i in range(k - 1):

        dist = []
        ### For loop gets distances of trajectories to nearest centroid
        for T in trajectories_list:
            min_distance = float('inf')
            for C in centers:
                distance, _ = dtw(T, C)
                if distance < min_distance:
                    min_distance = distance
            dist.append(min_distance)
        ### Calculate the weighted probabilitis to decide if a trajectory will become a centroid
        total = sum([d ** 2 for d in dist])
        probs = [d / total for d in dist]
        centers.append(random.choices(trajectories_list, weights=probs, k=1)[0])



    ### Add trajectories to cluster based off of which centroid they are closest to
    clusters = {}
    ### init clusters
    for i in range(k):
        clusters[i] = {}  # init dict of dicts

    for id in trajectory_ids:
        min_distance = float('inf')
        dists = [0 for i in range(k)]
        for i in range(k):

            dists[i] = dtw(centers[i],trajectories[id])

        indx = dists.index(min(dists))
        clusters[indx][id] = trajectories[id]

    return clusters


def random_seeding(trajectories, k):
    #random seeding function
    trajectory_ids = list(trajectories.keys())

    clusters = {} # key: int from 0 to k-1, value: dictionary of trajectory ids, where each id maps to trajectory list of points
    for i in range(k): 
        clusters[i] = {} # init dict of dicts

    for id in trajectory_ids:
        cluster_id = random.randint(0, k-1) # assign trajectory to random cluster
        clusters[cluster_id][id] = trajectories[id]

    return clusters



def dtw(P, Q):
    #dtw edited from case study part 1 to return distance along with path
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


def center_computation(cluster, n_ticks=20):
    # -- get x_min and x_max
    x_min, x_max = math.inf, -math.inf
    for id in cluster.keys():
        traj = cluster[id]
        for p in traj:
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]

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


def lloyds(TS, k, seed, max_iter):

    if seed == 'random':
        clusters = random_seeding(TS, k) # clusters is a dict of dicts of trajectories, indexed 0 to k-1
    elif seed == 'proposed':
        clusters = proposed_seed(TS, k) # clusters is a dict of dicts of trajectories, indexed 0 to k-1
    else:
        raise NotImplementedError
    
    clusters_old = clusters # keep record of old cluster to see if stopping condition is met
    centers = {} #key: indexed from 0 to k-1; val: center trajectory (list of tuples)
    counter = 0
    max_iter = 10 # CHANGE
    cost = 0

    while counter < max_iter-1:
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


def find_cost(clusters):
    cost = 0
    for i in clusters:
        centroid = center_computation(clusters[i])
        for id in clusters[i]:
            distance, _ = dtw(clusters[i][id], centroid)
            cost += distance
    return cost


def finding_random_seed_costs():
    #function to find random seed costs
    random_seed_costs = {}
    trajectories = read_file('geolife-cars-upd8.csv')

    costs_4 = []
    clusters_4 = lloyds(trajectories, 4, 'random', 10)
    for i in range(3):
        costs_4.append(find_cost(clusters_4))

    numer_4 = costs_4[0] + costs_4[1] + costs_4[2]
    random_seed_costs[4] = numer_4 / 3

    costs_6 = []
    clusters_6 = lloyds(trajectories, 6, 'random', 10)
    for i in range(3):
        costs_6.append(find_cost(clusters_6))

    numer_6 = costs_6[0] + costs_6[1] + costs_6[2]
    random_seed_costs[6] = numer_6 / 3

    costs_8 = []
    clusters_8 = lloyds(trajectories, 8, 'random', 10)
    for i in range(3):
        costs_8.append(find_cost(clusters_8))

    numer_8 = costs_8[0] + costs_8[1] + costs_8[2]
    random_seed_costs[8] = numer_8 / 3

    costs_10 = []
    clusters_10 = lloyds(trajectories, 10, 'random', 10)
    for i in range(3):
        costs_10.append(find_cost(clusters_10))

    numer_10 = costs_10[0] + costs_10[1] + costs_10[2]
    random_seed_costs[10] = numer_10 / 3

    costs_12 = []
    clusters_12 = lloyds(trajectories, 12, 'random', 10)
    for i in range(3):
        costs_12.append(find_cost(clusters_12))

    numer_12 = costs_12[0] + costs_12[1] + costs_12[2]
    random_seed_costs[12] = numer_12 / 3

    return random_seed_costs

def plot_random(costs_random):
    #plots the random seed costs
    x = list(costs_random.keys())
    y = list(costs_random.values())
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel('AVG Cost at K')
    plt.title('Cost of Lloyds with Random Seeding')
    plt.show()

def finding_proposed_seed_costs():
    #find the proposed seed costs
    proposed_seed_costs = {}
    trajectories = read_file('geolife-cars-upd8.csv')

    costs_4 = []
    clusters_4 = lloyds(trajectories, 4, 'proposed', 10)
    for i in range(3):
        costs_4.append(find_cost(clusters_4))

    numer_4 = costs_4[0] + costs_4[1] + costs_4[2]
    proposed_seed_costs[4] = numer_4 / 3
    print("done 4")
    costs_6 = []
    clusters_6 = lloyds(trajectories, 5, 'proposed', 10)
    for i in range(3):
        costs_6.append(find_cost(clusters_6))

    numer_6 = costs_6[0] + costs_6[1] + costs_6[2]
    proposed_seed_costs[6] = numer_6 / 3
    print("done 6")
    costs_8 = []
    clusters_8 = lloyds(trajectories, 8, 'proposed', 10)
    for i in range(3):
        costs_8.append(find_cost(clusters_8))

    numer_8 = costs_8[0] + costs_8[1] + costs_8[2]
    proposed_seed_costs[8] = numer_8 / 3
    print("done 8")
    costs_10 = []
    clusters_10 = lloyds(trajectories, 10, 'proposed', 10)
    for i in range(3):
        costs_10.append(find_cost(clusters_10))

    numer_10 = costs_10[0] + costs_10[1] + costs_10[2]
    proposed_seed_costs[10] = numer_10 / 3
    print("done 10")
    costs_12 = []
    clusters_12 = lloyds(trajectories, 12, 'proposed', 10)
    for i in range(3):
        costs_12.append(find_cost(clusters_12))

    numer_12 = costs_12[0] + costs_12[1] + costs_12[2]
    proposed_seed_costs[12] = numer_12 / 3
    print("done 12")
    return proposed_seed_costs

def plot_proposed(costs_proposed):
    #plots the proposed seed costs
    x = list(costs_proposed.keys())
    y = list(costs_proposed.values())
    plt.plot(x, y)
    plt.xlabel('K Value')
    plt.ylabel('Average Cost at K')
    plt.title('Cost of Lloyds with Proposed Seeding')
    plt.show()

def plot_iteration_cost_random():
    #plots iteration costs for random
    C_j = []
    runs = 5
    k = 4
    max_iterations = 4
    trajectories = read_file('geolife-cars-upd8.csv')

    for j in range(max_iterations):
        total = 0
        for i in range(runs):
            # For every run, calculate the cost at the jth iteration
            cost = find_cost(lloyds(trajectories, 4, 'random', max_iterations))
            print("ran an iter")
            total += cost
        C_j.append(1 / runs * total)

    plt.plot(C_j)
    plt.xlabel('Iteration')
    plt.ylabel('Average Cost')
    plt.title('Average Cost over Iterations for runs = 5')
    plt.show()

def plot_iteration_cost_proposed():
    #plots iteration costs for random
    C_j = []
    runs = 5
    k = 4
    max_iterations = 4
    trajectories = read_file('geolife-cars-upd8.csv')

    for j in range(max_iterations):
        total = 0
        for i in range(runs):
            # For every run, calculate the cost at the jth iteration
            cost = find_cost(lloyds(trajectories, 4, 'proposed', max_iterations))
            print("ran an iter")
            total += cost
        C_j.append(1 / runs * total)

    plt.plot(C_j)
    plt.xlabel('Iteration')
    plt.ylabel('Average Cost')
    plt.title('Average Cost over Iterations for runs = 5')
    plt.show()

def retrieve_centers(tracjectories, k,seed_type, max_iterations):
    all_centers = []
    trajectories = read_file('geolife-cars-upd8.csv')
    clusters = lloyds(trajectories, k, seed_type, max_iterations)

    for i in clusters.keys():
        all_centers.append(center_computation(trajectories[i]))

    print(len(clusters))

    return all_centers


if __name__ == '__main__':
    #plot_random(finding_random_seed_costs())
    #plot_proposed(finding_proposed_seed_costs())
    #plot_plot_iteration_cost_random()
    #plot_iteration_cost_proposed()
    pass