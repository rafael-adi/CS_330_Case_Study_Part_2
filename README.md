# CS 330: Case Study Part 2

Authors: Nate Zelter, Lucas Josephy, Will Yun, Rafael Adi 

## Task 4: Center Trajectories

#### Compilation Instructions

To compile and run this code, follow these instructions:

1. Install Python 3: This code is written in Python 3, so you will need to have Python 3 installed on your system.
2. Install required libraries: This code uses the following libraries:
- pandas
- numpy
- matplotlib
- scipy
3. To install these libraries, run the following command in your terminal:
- pip install pandas numpy matplotlib scipy
4. Download the code: Download the code from wherever it is hosted.
5. Run the code: In your terminal, navigate to the directory where the code is saved

#### Execution Instructions 

1. On the bottom of task4.py, uncomment which parts of the task you want to execute.
2. If you are creating a figure ensure that the figure window is closed before running the file again. Close the window 
to end the file execution.

#### Organization of Code

The code is a Python script that contains functions for computing dynamic time warping (DTW) distance between two 
time-series trajectories. It also includes helper functions for converting between trajectory and x-y coordinate 
representations and for computing Euclidean distance between points.

The dtw function implements the DTW algorithm along with an extension to compute the average distance between matched 
pairs of points in the two trajectories. This is done by maintaining two dynamic programming tables, one for assignment 
sizes and another for average distances. The function returns an array of indices that indicate which pairs of points in 
the two trajectories are matched, as well as the DTW distance and the Eavg assignment.

The script also defines several other helper functions for manipulating trajectories and computing distances, including 
dist, traj_to_xy, and xy_to_traj.

The next function is center_trajectory_method_1 takes a dictionary TS as input, where each key is a trajectory id and 
each value is a list of xy points as tuples. The function computes the sum of the distances between each trajectory and 
all the other trajectories in the set, and returns the trajectory with the smallest sum as the center trajectory.

The second function center_trajectory_method_2 also takes a dictionary TS as input, but it constructs the center 
trajectory using linear interpolation. It first determines the x-axis range of the trajectories and constructs a 
linspace for the x-axis. Then, it computes the y-values of each trajectory at the x-axis points using linear 
interpolation. Finally, it computes the average of the y-values across all trajectories at each x-axis point to obtain 
the y-values of the center trajectory. The center trajectory is returned as a list of (x, y) coordinate tuples.

## Task 5: Clustering Trajectories

#### Compilation Instructions

- Run python3 task5.py when you are in the local directory.
- Ensure that scipy is installed on your local host machine; if not, run pip3 install scipy.

#### Execution Instructions 

1. Install Python: This code is written in Python, so you'll need to have Python installed on your machine in order to 
run it. You can download the latest version of Python from the official website: https://www.python.org/downloads/
2. Install the required packages: This code uses several Python packages, including math, matplotlib, and csv. These 
packages should already be included with a standard Python installation, but if they're not, you can install them using 
the following commands in your terminal or command prompt:
- pip install math
- pip install matplotlib
- pip install csv
- pip install pandas
- pip install scipy
- pip install numpy
3. Download the data: This code uses a dataset called geolife-cars-upd8.csv, which contains trajectory coordinates for 
several car trips. You'll need to download this file and place it in the same directory as the Python code.
4. Run the code: Once you have Python and the required packages installed, and you have downloaded the geolife-cars.csv 
file and placed it in the same directory as the Python code, you can run the code by opening a terminal or command 
prompt and navigating to the directory containing the code and the data file. Then, enter the following command:
- python Task5.py
5. View the output: The code will generate several line plots. These plots will be displayed in separate windows. You can 
close the windows to view the next histogram.


#### Organization of Code

- The code is implementing a clustering algorithm for trajectory data using both a random seeding method and a K-means++ algorithm. The trajectory data
 is stored in a dictionary format, where the key represents the trajectory ID, and the value is a list of tuples, where 
 each tuple represents a point in the trajectory (x, y) coordinates.

The main function proposed_seed implements the K-means++ algorithm for clustering the trajectories, and it returns a 
dictionary of dictionaries. The key of the outer dictionary represents the cluster ID (which is an index from 0 through k-1, and the value is another 
dictionary, where the key is the trajectory ID, and the value is the list of points in that trajectory. The K-means++ 
algorithm is implemented in two parts. The first part initializes the cluster centroids using a probabilistic method to 
choose the initial centroids. The second part assigns the remaining trajectories to the nearest cluster based on the 
distance to the centroids.

The random_seeding function is an alternative method to initialize the centroids. It simply assigns each trajectory to a 
random cluster centroid.

The euclidean_distance function calculates the Euclidean distance between two points in 2D space.

The dtw function implements the dynamic time warping (DTW) algorithm to calculate the distance between two trajectories. 
The function returns the DTW distance and the optimal warping path between the two trajectories.

The center_computation function computes the x-axis center of the trajectory data. It finds the minimum and maximum 
x-axis values and returns the center of the interval between these two values.

Lloyd's algorithm repeatedly calls the center computation method and then proceeds to re-assign each trajectory to the center
it is the closest to. It repeats this until either the max iterations is reached, or the stop condition of the clusters being 
the same before and after an iteration. 

The cost function sums the difference of every trajectory and its center after lloyd's algorithm is finsihed running. 