# Define memory and prediction_window
memory = 2
prediction_window = 3

##################################
# Name:        Matthew S. Jones
# Course:      AI 530: Big Ideas in AI
# Instructor:  Dr. Houssam Abbas
# Assignment:  7a
# Due Date:    15 November 2024

# Import packages
import warnings
import array
from collections import defaultdict
warnings.simplefilter(action='ignore', category=FutureWarning)

# Remove punctuation and special chars
def remove_characters(string):
    
    # Punctuation and other special characters to be removed from words
    characters_to_remove = ",;:.!?\"“”()"

    cleanword = ''.join(c for c in string if c not in characters_to_remove)

    if cleanword.startswith("‘"):
                
        cleanword = cleanword[1:]
                
    if cleanword.endswith("’"):
                
        cleanword = cleanword[:-1]
        
    return cleanword
                
    
def build_model(memory, prediction_window):

    # Can't start until at least the number of words in the memory
    # plus the prediction window have been read
    buffer_size = memory + prediction_window
    wordcount = 0
    
    # The last "delay" number of words
    buffer = [""] * buffer_size
    
    # Keep a count of pairs to make it easier to compute the percentages later
    totalpairs = 0
    
    model = defaultdict(dict)
    
    
    with open("input.txt", 'r') as file:
        
        for line in file:
            
            for word in line.split():
                
                # Remove punctuation and special chars
                cleanword = remove_characters(word) 
                
                # Make the words all lower-case to avoid duplicates due to capitalization
                word = cleanword.lower()
                
                # Skips words that aren't to be included in the model 
                if word == "the" or word == "a" or word == "an" or \
                word == "this" or word == "that" or word == "":
                    continue 

                if word[0] == "—":
                
                    word = word[1:]
                
                # Updates the buffer by removing a word from the beginning and 
                # adding the new one to the end
                wordcount = wordcount + 1
                buffer.pop(0)
                buffer.append(word)
                
                # Only do this is the buffer is full
                if wordcount >= buffer_size:
                    
                    totalpairs = totalpairs + 1

                    # Splits the buffer into word1 and word2 based on values 
                    # of memory and prediction_window
                    
                    buff1 = buffer.copy()
                    buff2 = buffer.copy()
                    
                    i = 0
                    while i < memory:
                        
                        buff2.pop(0)
                        i = i + 1
                        
                    i = 0
                    while i < prediction_window:
                    
                        buff1.pop()
                        i = i + 1

                    word1 = ' '.join(buff1)
                    word2 = ' '.join(buff2)
                    pair = word1 + ";" + word2

                    # Increments the count if the pair exists, 
                    # sets it to one if it doesn't 
                    currcount = model.get(pair)
                    
                    if currcount:
                    
                        model[pair] = model[pair] + 1
                        
                    else:
                    
                        model[pair] = 1
                        
                    # print(pair, ":", model[pair])
                        
    # print ("Total pairs: ", totalpairs)
    file.close()
    
    for key, value in model.items():
        
        # print (key, ":", value)
        percentage = value / totalpairs
        model[key] = percentage

    modfile = "model.txt"
    f = open(modfile, "w")
        
    word1 = ""

    for key, value in sorted(model.items()):
        
        # print (key, ":", value)
        splitkey = key.split(';')
        
        # Creates a new section if word1 changes
        if splitkey[0] != word1:
            
            word1 = splitkey[0]
            addplus = "+" + word1
            print (addplus, file=f)
            
        print (splitkey[1], value, file=f)
    
    f.close()    

    
# Call the model builder function
build_model(memory, prediction_window)
