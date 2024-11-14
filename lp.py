# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
m = 15 # Products
n = 10 # Components
# Aij amount of n/i required to build each m/j 

np.random.seed(1)              # ensures the same random numbers appear each time
s0 = np.random.randn(m)        # Creates a random array of size m
lamb0 = np.maximum(-s0, 0)     # Not sure what this does...
s0 = np.maximum(s0, 0)         # Ensures the number is positive?
x0 = np.random.randn(n)        # Creates a random array of size n
A = np.random.randn(m, n)      # Creates a random matrix of size m by n
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),   # minimizing this mean maximizing profit
                 [A @ x <= b])          # keeping this <= b means didn't exceed component budget
prob.solve(solver="ECOS")
# Added solver="ECOS" to override the default and avoid a license issue

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)