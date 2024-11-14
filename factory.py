##################################
# Name:       Matthew S. Jones
# Course:     AI 530: Big Ideas in AI
# Instructor: Dr. Houssam Abbas
# Assignment: 2
# Due Date:   11 October 2024

# Instructions:
# Recall there are three variables, x1, x2 and x3. 
# The objective is f(x) = x1*(S1-C1) + x2*(S2-C2)+x3*(S3-C3). 
# The constraints are x1 >= 0, x2 >= 0, x3 >= 0, x1*C1 + x2*C2 + x3*C3 <= 3,000,000.
# Use the following values: S1 = 100, S2 = 200, S3 = 300, C1 = 50, C2 = 55, C3 = 75.
# Solve the problem a second time, but with parameter values S1 = 50, S2 = 201, S3 = 200, C1 = 52.4, C2 = 55, C3 = 75.

# Import packages.
import cvxpy as cp
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def optcpu (m, S, C):

    np.random.seed(1)         # ensures the same random numbers appear each time
    X = np.random.random(m)   # Creates a random array of size m; will represent the number of each CPU produced

    # Define and solve the CVXPY problem.
    X = cp.Variable(m)

    Y = X.T @ S - X.T @ C     # Calculates the objective function (profit)
    Z = X.T @ C               # Calculates the constraint (budget)

    prob = cp.Problem(cp.Maximize(Y),  # maximizing this mean maximizing profit
                      [Z <= 3000000,   # keeping this <= 300k means spending doesn't exceed the budget
                       X >= 0])        # ensures that CPU production is positive
    prob.solve(solver="ECOS")          # Added solver="ECOS" to override the default and avoid a license issue

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution X is")
    print(X.value)

m = 3 # different CPUs

# Use the following values: S1 = 100, S2 = 200, S3 = 300, C1 = 50, C2 = 55, C3 = 75.
S = [100, 200, 300] # Selling price of each CPU 
C = [50, 55, 75]    # Cost to produce each CPU 

print ("\nSelling Prices: ", S)
print ("Production Costs: ", C)

optcpu(m, S, C)

# Solve the problem a second time, but with parameter values S1 = 50, S2 = 201, S3 = 200, C1 = 52.4, C2 = 55, C3 = 75.
S = [50, 201, 200] # Selling price of each CPU 
C = [52.4, 55, 75]    # Cost to produce each CPU 

print ("\nSelling Prices: ", S)
print ("Production Costs: ", C)

optcpu(m, S, C)
