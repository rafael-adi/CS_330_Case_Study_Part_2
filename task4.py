"""
Task 4 for CS 330 Case Study
task4.py

April 2023
NZ and LJ
"""

import math

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


if __name__ == '__main__':
    main()