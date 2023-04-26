import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math


IDS = ["115-20080527225031",
        "115-20080528230807",
        "115-20080618225237",
        "115-20080624022857",
        "115-20080626014331",
        "115-20080626224815",
        "115-20080701030733",
        "115-20080701225507",
        "115-20080702225600",
        "115-20080706230401",
        "115-20080707230001"]

def dist(p1, p2):
    # Compute the distance between two points
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def traj_to_xy(trajectory):
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]
    return x, y


def xy_to_traj(x, y):
    assert len(x) == len(y)
    return [(x[i], y[i]) for i in range(len(x))]


def dtw_new(P, Q):
    # Nate's version: O(n1 * n2)
    n1 = len(P)
    n2 = len(Q)
    cost = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost[i,j] = dist(P[i], Q[j])

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


def dtw_raf(P, Q):
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



def center_trajectory_method_1(TS):
    """
    computes center trajectory over a trajectory set TS
    _id = trajectory id, a string
    _literal = trajectory literal, a list of xy points as tuples
    """

    ids = list(TS.keys())
    m = math.inf
    Tc = ''

    for T_id in ids:
        s = 0
        for Tprime_id in ids: 
            if Tprime_id == T_id: # skip if same trajectory
                continue

            T_literal = TS[T_id] 
            Tprime_literal = TS[Tprime_id]
            delta, dtw_T_Tprime = dtw_new(T_literal, Tprime_literal)
            s += delta

        if s < m: # ith sum is the less than current global minumum sum
            m = s
            temp = []
            for i, j in dtw_T_Tprime:
                temp.append((T_literal[i][0], Tprime_literal[j][1])) # construct trajectory from dtw path
            Tc = temp

    return Tc



def center_trajectory_method_2(TS):

    # -- gather x_min and x_max for linspace bounds
    x_min, x_max = -math.inf, math.inf
    for id in TS.keys():
        if TS[id][-1][0] > x_min:
            x_min = TS[id][-1][0]
        if TS[id][0][0] < x_max:
            x_max = TS[id][0][0]

    print(x_min, x_max)

    # -- make linspace for x axis
    n_ticks = 100
    x_ticks = np.linspace(x_min, x_max, n_ticks)   
    y_values = np.zeros((len(TS), n_ticks))
    for i, traj in enumerate(TS):
        # Separate x and y values from the trajectory
        x_vals = np.array([p[0] for p in TS[traj]])
        y_vals = np.array([p[1] for p in TS[traj]])
        
        # Use linear interpolation to find y values at the x ticks
        interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate')
        y_values[i] = interp_func(x_ticks)
    
    # Average the y values across all trajectories at each x tick to construct the center trajectory
    center_y = np.mean(y_values, axis=0)
    # print(y_values)
    
    # Construct the center trajectory as a list of (x, y) coordinate tuples
    center_traj = [(x_ticks[i], center_y[i]) for i in range(n_ticks)]

    return center_traj


    

def main():

    # -- read in a data set from csv
    filename = 'geolife-cars-upd8.csv'
    # filename = 'geolife-cars-sixty-percent.csv'
    df = pd.read_csv(filename)

    # -- gather all trajectories in data set
    # df.sort_values(by=["id_", "date"], inplace=True)
    trajectories = {}   
    for index, row in df.iterrows():
        id = row["id_"]
        # timestamp = row["date"] # not being used rn
        x = row["x"]
        y = row["y"]
        
        if id not in trajectories:
            trajectories[id] = []
        
        # trajectories[id].append((x, y, timestamp)) # with timestamps
        trajectories[id].append((x, y)) # without timestamps


    # -- gather important trajectory ids
    TS = {}
    for id in trajectories.keys():
        if id in IDS:
            TS[id] = trajectories[id]

    # print(TS.keys())

    center_traj1 = center_trajectory_method_1(TS)
    center_traj2 = center_trajectory_method_2(TS)
    # print(center_traj2)

    # print(center_traj1)
    # print(center_traj2)


    # -- visualize trajectories
    fig, ax = plt.subplots()
    # ax.plot(x_ticks, y_values[0], 'o', color='black', label="Linearly Interpolated Points")
    # ax.plot(x_ticks, y_values[1], 'o', color='black')
    for id in TS.keys():
        ax.plot([p[0] for p in TS[id]], [p[1] for p in TS[id]], label=id)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory Centering Approaches I and II")
    # plt.show()
    
    # print(center_traj)
    ax.plot([p[0] for p in center_traj1], [p[1] for p in center_traj1], linestyle='--', label='Approach I Center Trajectory')
    ax.plot([p[0] for p in center_traj2], [p[1] for p in center_traj2], linestyle='--', label='Approach II Center Trajectory')
    ax.legend()
    ax.grid()
    plt.show()

    for id in TS.keys():
        dist, path = dtw_new(center_traj1, TS[id])
        print(str(id) + " to Approach I Center Trajectory: " + str(dist))
    for id in TS.keys():
        dist, path = dtw_new(center_traj2, TS[id])
        print(str(id) + " to Approach II Center Trajectory: " + str(dist))


if __name__ == "__main__":
    print("starting task4...")
    main()