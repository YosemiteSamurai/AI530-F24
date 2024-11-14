##################################
# Name:       Matthew S. Jones
# Course:     AI 530: Big Ideas in AI
# Instructor: Dr. Houssam Abbas
# Assignment: 5
# Due Date:   1 November 2024

# Instructions:
# k-nearest neighbors (k-NN) is a very popular and long-used algorithm for supervised learning. Given a set of labeled data points, the objective is to determine the label of a new, test, point. k-NN does that by first finding the k points in the data set that are nearest the test point. It then determines which label appears most often in these k nearest neighbors. The test point is assigned that label.
# For instance, if k=3, and the 3 nearest points have labels Red, Red and Orange, the test point is labeled Red. 
# The basic assumption is that near-by points probably share the same/similar labels. This can be seen as an assumption of label "continuity" in the underlying metric.
# Now suppose all the data points live in the unit cube C(d) in d dimensions (so that's the cube of side length 1 and with a corner at 0).
# Write a piece of code that will generate N data points uniformly at random in C(d).
# Then compute all pair-wise distances between the points (i.e., the distances ||pi - pj|| between all pairs of points pi and pj).
# Plot a histogram of these values. The x-axis range in all historgrams should be the same. 
# Report  the variance of the distances.
# Repeat this for values of d = 2, 3, 10, 100, 1000, 10000.
# What do you notice as d grows? What does this mean for the performance of k-nearest neighbors in high dimensions?
# Leave N and d as variables that can be set, don't hard-code them. For generating your experiments, set N=200. 

# Import packages
import random
import warnings
import numpy as np 
from math import dist
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define constants
N = 400
d = 10000
print ("N: ", N)
print ("d: ", d)
outfile = "knn.txt"
f = open(outfile, "w")

# Initialize matrix with random numbers
cubedata = [[0 for _ in range(d)]]*N
np.random.seed(99)
cubedata = np.random.uniform(0.0, 1.0, size = (N,d))

# Calculate distances and write to a file

i = 0

while i < N:
    
    j = i + 1
    
    while j < N:
        
        distij = dist(cubedata[i], cubedata[j])
        # print ("i, j, cubedata[i], cubedata[j], distij", i, j, cubedata[i], cubedata[j], distij)
        print (distij, file=f)
        
        j = j + 1
    
    i = i + 1
    
f.close()
