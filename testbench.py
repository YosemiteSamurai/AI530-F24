import numpy as np
from tsp import tsp

# Expect 0, 4, 3, 2, 1, 0 with cost 7 (requires backtracking)
M40 = np.array(
[[-1, 1, 1, -1, 2],
[1, -1, 1, 4, -1],
[1, 1, -1, 2, 1],
[-1, 4, 2, -1, 1],
[2, -1, 1, 1, -1],
])

# Expect 0,1,3,2,0 with cost 7 (no backtracking necessary)
M41 = np.array(
[[-1, 1, 2, 5],
[1, -1, 2, 1],
[2, 2, -1, 3],
[5, 1, 3, -1]])

# No tour because 6 forces us to re-visit 5
M42 = np.array(
[
[-1, 1, 1, -1, 1, -1],
[1, -1, 1, 4, -1, -1],
[1, 1, -1, 2, 1, -1],
[-1, 4, 2, -1, 1, -1],
[1, -1, 1, 1, -1, 1],
[-1, -1, -1, -1, 1, -1],
])

print("Test cases")
print("t40")
t40 = tsp(M40, 50)
print(t40)
print("t41")
t41 = tsp(M41, 50)
print(t41)
print("t42")
t42 = tsp(M42,50)
print(t42)