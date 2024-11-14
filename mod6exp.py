##################################
# Name:        Matthew S. Jones
# Course:      AI 530: Big Ideas in AI
# Instructor:  Dr. Houssam Abbas
# Exploration: 6
# Due Date:    8 November 2024

# Import packages
import random
import warnings
import numpy as np 
import math 
import array
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def NN(x1, x2, w1, w2):
    wghtavg = (w1 * x1) + (w1 * x2)
    return sigmoid(wghtavg)

def G(x1, x2):
    return math.exp((x1**2+x2**2)/-2)/math.sqrt(2*math.pi)

def geterr(w1, w2):
    
    x1 = []
    x2 = []
    y = []
    z = []
    e = 0

    i = -1

    while i <= 1:

        j = -1
    
        while j < 1:
        
            k = NN(i, j, w1, w2)
            l = G(i, j)
        
            err = abs(k - l)
            e = e + err
        
            x1.append(i)
            x2.append(j)
            y.append(k)
            z.append(l)
        
            j = j + 0.1
        
        i = i + 0.1
        
    return e

minerr = 10000
minw1 = 0
minw2 = 0

a1 = []
a2 = []
c = []
rng = 0.5

i2 = -1 * rng

while i2 <= rng:

    j2 = -1 * rng
    
    while j2 < rng:
        
        e1 = geterr(i2, j2)
        
        if e1 < minerr:
            
            minerr = e1
            minw1 = i2
            minw2 = j2
        
        a1.append(i2)
        a2.append(j2)
        c.append(e1)
        
        j2 = j2 + 0.01
        
    i2 = i2 + 0.01
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a1,a2,c)
plt.show() 

print ("Min Error: ", minerr)
print ("w1, w2: ", minw1, minw2)