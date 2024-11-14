# This program only needs the prompt; it will get memory and
# prediction_window from the model file

# Define the prompt
prompt = "After dinner"

# Define the number of words in the output    
length = 50

##################################
# Name:        Matthew S. Jones
# Course:      AI 530: Big Ideas in AI
# Instructor:  Dr. Houssam Abbas
# Assignment:  7b
# Due Date:    15 November 2024

# Import packages
import warnings
import array
import math
import random
import numpy as np 
from collections import defaultdict
warnings.simplefilter(action='ignore', category=FutureWarning)

# Reads the model and returns the memory and prediction_window values
def read_model():

    with open("model.txt", 'r') as file:
        
        for line in file:

            # Remove the newline character from the line            
            line = line.strip()
            
            # If it starts with a + then it must be a new word1
            if line[0] == "+":
                
                # Determine word1 and initialize the
                # percentage and hash value
                word1 = line[1:]
                percentage = 0
                model[word1] = ""

                # Determine the memory from the model file                
                splitw1 = word1.split(' ')
                memory = len(splitw1)

            else:

                # Determine the current percentage and word2
                splitline = line.split(' ')
                currper = float(splitline.pop())
                word2 = ' '.join(splitline)
                
                # Determine the prediction_window from the model file
                splitw2 = word2.split(' ')
                prediction_window = len(splitw2)

                # Update the most-common word if the
                # percentage is greater than what was                
                if currper > percentage:
                    
                    model[word1] = word2
                    percentage = currper
 
                # If the percentage is the same then 
                # append it the existing value 
                elif currper == percentage:
                    
                    model[word1] = model[word1] + ";" + word2

    file.close()
    
    return (memory, prediction_window)

# Returns the length of an array that was initialized to ""    
def mylen(myarray):
    
    i = 0
    
    while myarray[i] != "":
    
        i = i + 1
        
    return i

# Picks a random word2 when there is more than one
# with the top percentage    
def pickrand(nextwords):
    
    splitnext = nextwords.split(';')
    
    if len(splitnext) == 1:
        
        return splitnext[0]
        
    else:
    
        i = random.randint(1, len(splitnext)) - 1
        return splitnext[i]
    
def run_model(memory, prediction_window, prompt):
    
    output = [""] * ((math.ceil(length / prediction_window) \
    * prediction_window) + prediction_window)
    
    i = 0
    while i < memory:
    
        output[i] = splitprompt[i]
        i = i + 1

    while output[length] == "":
        
        startprev = mylen(output) - memory
        prevarr = output[startprev:startprev+memory]
        prevword = ' '.join(prevarr)

        # If more than one word2 was the highest percentage
        # in the model, then randomly pick one        
        nextword = pickrand(model[prevword])
        splitnext = nextword.split(' ')

        i = mylen(output)
        count = 0
        
        while count < prediction_window:
            
            output[i] = splitnext[count]
            i = i + 1
            count = count + 1

    # trim output down to length
    newoutput = output[:length]
    
    sentence = ' '.join(newoutput)
    print ("Output:", sentence)


#########################################################################
    
# Read the model from the text file
model = defaultdict(dict)
(memory, prediction_window) = read_model()

print ("Memory:", memory)
print ("Prediction Window:", prediction_window)
print ("Prompt:", prompt)
print ("Length:", length)
    
# make the words all lower-case to avoid duplicates due to capitalization
prompt = prompt.lower()
    
splitprompt = prompt.split(' ')
promptlen = len(splitprompt)
    
# Make sure that the prompt is consistent with memory!!
if promptlen != memory:
        
    print ("\nError: The number of words in the prompt \
    must match the number of words in the model memory.")

# Make sure the prompt is in the model
elif prompt in model.keys(): 
    
    run_model(memory, prediction_window, prompt)
    
else: 
    
    print ("\nError: The given prompt is not found in the model, sorry!")
    