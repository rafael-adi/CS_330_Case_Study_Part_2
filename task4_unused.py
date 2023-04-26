"""
Task 4 for CS 330 Case Study
task4.py

April 2023
NZ and LJ
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def centroid(trajectory, avg_distance=False):
    trajectory = np.array(trajectory) # O(n)
    centroid = trajectory.mean(axis=0) # O(n)
    if avg_distance: # O(n^2)
        distances = np.linalg.norm(trajectory - centroid, axis=1)
        average_distance = np.mean(distances)
        return centroid, average_distance
    return centroid


def d(q, e):
    # Compute the distance between a point q and a segment e

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
    dp = (ab_x*aq_x) + (ab_y*aq_y)
    abLen = abs(math.sqrt((ax-bx)**2 + (ay-by)**2))
    q_to_AB = dp / (abLen**2)
    if (q_to_AB >= 1): # closest point is B
        return abs(math.sqrt((qx-bx)**2 + (qy-by)**2)) 
    elif (q_to_AB <= 0): # closest point is A
        return abs(math.sqrt((qx-ax)**2 + (qy-ay)**2)) 
    else: # closest point is in the middle of line segment e, let f be that point
        fx = (ab_x*q_to_AB) + ax
        fy = (ab_y*q_to_AB) + ay 
        return abs(math.sqrt((qx-fx)**2 + (qy-fy)**2))


def greedy(T, ep=0.3):
    # Implement the greedy algorithm TS-greedy(T,ε) to compute an ε-simplification of T.

    traj = [T[0], T[-1]]
    for i in range(1, len(T)-1):
        dist = d(T[i], [T[0],T[-1]])
        if dist > ep:
            left_traj = greedy(T[:i+1], ep)
            right_traj = greedy(T[i:], ep)
            traj = left_traj[:-1] + right_traj
            break
    return traj




def ct_method_1(TS):
    ids = list(TS.keys())

    m = math.inf
    Tc_id = ''
    for T_id in ids:
        s = 0

        for Tprime_id in ids:
            if Tprime_id == T_id:
                continue

            T_literal = TS[T_id]
            Tprime_literal = TS[Tprime_id]
            dtw_T_Tprime = dtw(T_literal, Tprime_literal)
            s += len(dtw_T_Tprime) #! this is boof

        if s < m: # ith sum is the less than current global minumum sum
            m = s
            Tc_id = Tprime_id




def ct_method_2(TS):

    """
    Find the centroid for each trajectory T in TS
    Construct new trajectory from list of centroids
    Return centroid of constructed trajectory
    """
                # P_array = np.array(P)
                # c_P = P_array.mean(axis=0)

    ids = list(TS.keys())

    

    centroids = []
    for T_id in ids:
        T_literal = TS[T_id]
        centroids.append(centroid(T_literal))
    Tc = centroid(centroids)


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

def fd(P, Q):
    lenP = len(P)
    lenQ = len(Q)

    #Initializing first dp table for maxes
    dp = [[math.inf] * (lenQ+1) for _ in range(lenP+1)]
    dp[0][0] = 0

    #Creating the table that will be used to find path later
    for i in range(1, lenP+1):
        for j in range(1, lenQ+1):
            distance = math.dist(P[i - 1], Q[j - 1])
            temp1 = dp[i-1][j-1]
            temp2 = dp[i - 1][j]
            temp3 = dp[i][j-1]
            mintemp = min(temp1, temp2, temp3)
            dp[i][j] = max(distance, mintemp)

    #Initializing Emax array for indices
    Emax = []

    i = lenP
    j = lenQ

    # Reverse looping over dp to add indices to Emax array (starting from bottom right corner)
    while i > 0 and j > 0:
        temp1 = dp[i - 1][j - 1]
        temp2 = dp[i - 1][j]
        temp3 = dp[i][j - 1]
        mintemp = min(temp1, temp2, temp3)
        if dp[i-1][j] == mintemp:
            Emax.append([i - 1, j])
            i -= 1
        elif dp[i-1][j-1] == mintemp:
            Emax.append([i - 1, j-1])
            i -= 1
            j -= 1
        else:
            Emax.append([i, j - 1])
            j -= 1
    return Emax[::-1]


def main():
    print("Hello world, task 4 here we go...")

    # -- read in a data set from csv
    filename = 'geolife-cars-ten-percent.csv'
    df = pd.read_csv(filename)

    # -- gather trajectories
    df.sort_values(by=["id_", "date"], inplace=True)
    trajectories = {}   
    for index, row in df.iterrows():
        id = row["id_"]
        timestamp = row["date"]
        x = row["x"]
        y = row["y"]
        
        if id not in trajectories:
            trajectories[id] = []
        
        trajectories[id].append((x, y, timestamp))
        # trajectories[id_].append({"x": x, "y": y, "timestamp": timestamp})

    # -- print some stuff
    n = len(trajectories) # number of trajectories in Trajectory Set
    print(f"Number of trajectories in Trajectory Set: {n}\n")
    ids = list(trajectories.keys())
    # print(f"IDs of trajectories in Trajectory Set: {ids}\n") #926 ids for 10% of data

    id_P, id_Q, id_R = ids[0], ids[1], ids[2]
    # P, Q, R = trajectories[id_P], trajectories[id_Q], trajectories[id_R]
    P = [(p[0], p[1]) for p in trajectories[id_P]]
    Q = [(q[0], q[1]) for q in trajectories[id_Q]]
    R = [(r[0], r[1]) for r in trajectories[id_R]]

    print(f"Trajectory P of length {len(P)}: {P}\n") #52 points for ids[0]
    print(f"Trajectory Q of length {len(Q)}: {Q}\n") #192 points for ids[1]
    print(f"Trajectory R of length {len(R)}: {R}\n") #53 points for ids[2]

    # -- test distance for P and Q
    # dist_PQ = dtw(P, Q)
    # # print(f"DTW distance between P and Q: {dist_PQ}\n") # len is 192

    # # -- test distance for P and R
    # dist_PR = dtw(P, R)
    # # print(f"DTW distance between P and R: {dist_PR}\n") # len is 53

    # # -- test distance for Q and R
    # dist_QR = dtw(Q, R)
    # # print(f"DTW distance between Q and R: {dist_QR}\n") # len is 53 


    # -- strip timestmaps from trajectories
    TS = {} # this is fancy T
    for id in ids:
        traj = trajectories[id]
        T = [(p[0], p[1]) for p in traj]
        TS[id] = T

    # -- trajectory simplification!!! 
    # todo for part 2

    # -- implement distance function delta(T, T')
    P, Q, R = TS[id_P], TS[id_Q], TS[id_R]
    # print(f"Trajectory P of length {len(P)}: {P}\n") #52
    # print(f"Trajectory Q of length {len(Q)}: {Q}\n") #192
    # print(f"Trajectory R of length {len(R)}: {R}\n") #53

    ### Method 2:
    """
    Find the centroid for each trajectory T in TS
    Construct new trajectory from list of centroids
    Return centroid of constructed trajectory
    """
    centroids = []
    for T_id in ids:
        T_literal = TS[T_id]
        centroids.append(centroid(T_literal))
    Tc = centroid(centroids)
    print(f"Centroid of Tc: {Tc}\n")

    # SANITY CHECK
  

                # P_array = np.array(P)
                # c_P = P_array.mean(axis=0)
                # fig, ax = plt.subplots()
                # ax.plot(P_array[:, 0], P_array[:, 1], label="P")
                # ax.scatter(c_P[0], c_P[1], c="red", label="c_P")
                # ax.legend()
                # ax.set_xlabel("X")
                # ax.set_ylabel("Y")
                # ax.set_title("Trajectory P with Centroid c_P")
                # plt.show()




    # -- find center trajecctory using Method 1 
    # result_1 = ct_method_1(TS)

    

if __name__ == '__main__':
    main()