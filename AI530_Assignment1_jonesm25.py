##################################
# Name:       Matthew S. Jones
# Course:     AI 530: Big Ideas in AI
# Instructor: Dr. Houssam Abbas
# Assignment: 1
# Due Date:   4 October 2024

# Instructions:

# Create a function called invertit.
# Function invertit accepts a matrix A.
# It checks whether A is invertible, and if it is invertible, it returns its inverse.
# If it is not invertible, it returns the empty array. 
# To check whether A is invertible, it first checks whether it is square.
# If it is not, then it is not invertible.
# If it is square, it computes the determinant. Matrix A is invertible if and only if its determinant is non-zero.
# For computing the determinant you may use a built-in function.
# Next, create a script called testinvert. The script generates 10 random matrices, half of which are square and the rest are not.
# It calls invertit on each one, and prints a message "Matrix is invertible" or "Matrix is not invertible". 
# Whatever is not explicitly specified, you should make a decision on how to set it. (E.g. we did not specify the size of the matrices).

import random 

# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy

def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant


# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy#:~:text=I%20want%20to%20invert%20a%20matrix

def eliminate(r1, r2, col, target=0):
    fac = (r2[col]-target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]

def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i+1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i+1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a

def inverse(a):
    tmp = [[] for _ in a]
    for i,row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0]*i + [1] + [0]*(len(a)-i-1))
    gauss(tmp)
    return [tmp[i][len(tmp[i])//2:] for i in range(len(tmp))]


def invertit(m):

    print ("Matrix =", m)

    rows = len(m)
    cols = len(m[0])

    # print ("Rows: ",rows)
    # print ("Cols: ",cols)
    
    m_inv = []

    if rows != cols: print ("Matrix is not invertible (not square)")

    else: 
        # print ("Matrix is square!")
        m_det = getMatrixDeternminant(m)
        # print ("Determinant = ", m_det)
    
        if m_det != 0:
        
            print ("Matrix is invertible")
            # m_inv = getMatrixInverse(m)
            m_inv = inverse(m)
            return m_inv
            
        else:
        
            print ("Matrix is not invertible (det is zero)")

    return m_inv
    

def testinvert():
    
    print ("Demonstrate five examples of square matrices:")
    print ("")
    for a in range (0, 5):

        m_size = random.randint(2, 4)
        # print ("Square: ", m_size)
    
        m = []
    
        for x in range (0, m_size):
            row = []
        
            for y in range (0, m_size):
                row.append(random.randint(1, 10))
    
            m.append(row)
        
        m_inv = invertit(m)    
        print ("Inverse = ", m_inv)
        print  ("")

    print ("Demonstrate what happens when the determinant is zero, in case that doesn't happen naturally:")
    print ("")
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m_inv = []
    invertit(m)
    print ("Inverse = ", m_inv)
    print ("")

    print ("Demonstrate five examples of non-square matrices:")
    print ("")
    for a in range (0, 5):

        m_size = random.randint(2, 4)
        # print ("Square: ", m_size)
    
        m = []
    
        for x in range (0, m_size):
            row = []
        
            for y in range (0, m_size+1):
                row.append(random.randint(1, 10))
    
            m.append(row)
        
        m_inv = invertit(m)    
        print ("Inverse = ", m_inv)
        print ("")

testinvert()
