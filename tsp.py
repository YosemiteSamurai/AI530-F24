##################################
# Name:       Matthew S. Jones
# Course:     AI 530: Big Ideas in AI
# Instructor: Dr. Houssam Abbas
# Assignment: 3
# Due Date:   18 October 2024

# Instructions:
# M is an n x n numpy array. K is an integer. optimal_tour is a list or array, and optimal_value is a float. 
# M is the matrix of distances; K is the maximum number of times to randomly go back and try again before giving up.
# Function tsp implements a randomized greedy search
# The search will not be complete, meaning that it might miss the minimal tour, but hopefully not by much.

# Import packages
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def checkpossible (M):
    
    numcities = len(M) # Get the number of cities on the tour
    
    possible = 1
    i = 0
    
    # loop for each city
    while i < numcities and possible == 1:
        
        j = 0
        negcount = 0
        
        # loop for each path in/out of the city 
        while j < numcities:
            
            if M[i][j] == -1:
                
                # count how many non-paths the city has 
                negcount = negcount + 1
                
            j = j + 1
        
        # if the city has only one path in/out then a tour is not possible
        if negcount == numcities - 1:
            
            possible = 0
            
        i = i + 1
    
    return possible

def tsp (M, K):

    numcities = len(M) # Get the number of cities on the tour
    
    # Initialize the tour as all -1 to indicate that a stop hasn't been chosen
    # This array is one element longer because we have to get back to city1 (0) to finish
    tour = [-1] * (numcities + 1)
    
    # Initialize a matrix to keep track of paths that have been tried to avoid repeating upon backtracking
    tries = [[0 for x in range(numcities)] for x in range(numcities)]

    # Initialize the number of backtracks; keep count and quit at K
    backcount = 0
    
    # use this array to keep track of the path lengths; initialize to 0
    pathlen = [0] * (numcities + 1)
        
    # Check whether a tour is possible before continuing
    possible = checkpossible(M)

    if possible == 1:

        # use this array to indicate which cities have been visited; 0 = No, 1 = Yes
        visited = [0] * numcities

        currstop = 0 # Start the tour at citi1
        stopcnt = 1  # Do as many stop as there are cities
        tour[0] = 0  # Stop 0 is city1, which is represented by 0

        # Need to make sure it gets back to city1 at the end, so the last new city MUST have a path to 1
        visited[0] = 1 # city1 has been visited
        
        while stopcnt < numcities and backcount < K:
        
            # Look for the lowest value at M[stop][j] where visited[j] = 0
        
            lowj = -1     # the current city (cc) with the lowest path length 
            shortest = -1 # the value of the shortest path length 
            j = 1         # don't need to check city1 since we started there
            
            while j <= numcities - 1:
                
                # Should pass if there is a path, it hasn't been tried, and the city hasn't been visited
                if M[currstop][j] != -1 and tries[currstop][j] == 0 and visited[j] == 0:
                
                    # Should pass if a short path hasn't been found or the current path is shorter
                    if shortest == -1 or M[currstop][j] < shortest:    
                        lowj = j
                        shortest = M[currstop][lowj]

                j = j + 1
                
            # After the the loop, check whether a path was found            
            # If a path was found, then update the path length and the tour, and mark the path as tried
            if shortest > -1:
            
                visited[lowj] = 1
                tour[stopcnt] = lowj 
                pathlen[stopcnt] = shortest
                tries[currstop][lowj] = 1
                currstop = lowj
                stopcnt = stopcnt + 1
                
                # if it's the last stop, make sure there's a path back to city1
                if stopcnt == numcities:
                    
                    if M[currstop][0] > 0:
                        
                        pathlen[stopcnt] = M[currstop][0]
                        tries[currstop][0] = 1
                        tour[stopcnt] = 0
                        stopcnt = stopcnt + 1
                        
                    # If a path back to city1 isn't possible, then go back to a random previous stop and try again             
                    else:
                        
                        # Have to go back at least two steps, but don't want to go back to zero
                        if stopcnt > 3:
                            
                            backto = np.random.randint(stopcnt-2)
                            backto = backto + 1
                           
                        else:
                            
                            backto = 1
                           
                        # print ("Back to: ", backto) 
                        stopcnt = backto
                        currstop = tour[backto-1]
                        
                        # reset which stops have been made to the goback point
                        while backto < numcities:
 
                            visited[tour[backto]] = 0
                            tour[backto] = -1
                            pathlen[backto] = 0
                            backto = backto + 1
                    
            # If a path was NOT found, then go back to a random previous stop and try again             
            else:             

                # Have to go back at least two steps, if possible, but don't want to go back to zero
                if stopcnt > 3:
                            
                    backto = np.random.randint(stopcnt-2)
                    backto = backto + 1
                           
                else:
                            
                    backto = 1
                           
                stopcnt = backto
                currstop = tour[backto-1]
                        
                # reset which stops have been made to the goback point
                while backto < numcities:
 
                    visited[tour[backto]] = 0
                    tour[backto] = -1
                    pathlen[backto] = 0
                    backto = backto + 1

            # Increment backcount to avoid an infinite loop if there's no solution
            backcount = backcount + 1
  
    else:
        
        # This means that at least one of the cities has only one path in/out 
        print ("No tour is possible, sorry!")
        totalpath = 0

    if backcount < K:
    
        totalpath = 0
        currstop = 0
    
        # Calculates the total path length based on the length between each pair of stops
        while currstop < numcities + 1:
        
            totalpath = totalpath + pathlen[currstop]
            currstop = currstop + 1
            
    else:
        
        # This means that it kept trying and never found a solution
        # This shouldn't happen for any of the testcases, obviously
        print ("No tour was found!")
        tour = [-1] * (numcities + 1)
        totalpath = 0
    
    return tour, totalpath