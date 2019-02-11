
# coding: utf-8

# # Mapping values from xml file

# In[1]:


import numpy as np
import pandas as pd
import xml.etree.ElementTree as et

#parse the xml file of the locations
tree = et.parse("/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/devset_topics.xml")
#get the root tag of the xml file
doc = tree.getroot()

mapping = {}
#map the location id(number) with the location name
for topic in doc:
    mapping[topic.find('number').text] = topic.find('title').text
    
print(mapping)
 


# # Create Vocabulary of all terms

# In[2]:


import numpy as np
import pandas as pd

#open the devset_textTermsPerPOI.wFolderNames file
f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerPOI.wFolderNames.txt', "r")
temp = []
vocab = []
key = 0

for line in f:
    b = line.split()
    x = b[0].count("_")
    temp.append(b[x+2::4])

# print(temp)
for x in temp:
    for y in x:
        vocab.append(y)

vocab = sorted(list(set(vocab)))


# # Get the LocationID, Model and value of k

# In[3]:


LocationID = input("Enter the Location id:")

Location = mapping[LocationID]

model = input("Enter the model: ")
if model == "TF":
    mod = 1
elif model == "DF":
    mod = 2
else:
    mod = 3
k = input("Enter value of k: ")


# # term frequency vector creation

# In[4]:


f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerPOI.wFolderNames.txt', "r")

locations_row = []
locationIds = []
location_temp = []
each_location_terms = []
terms = []
tf_values = []

#read file and get the location data in a list
for line in f:
    b = line.split()
    locations_row.append(b[0:])

location_terms = []
location = []
freqList = []

#get the locationIds of all the locations and store in locationIds list
for each_location in locations_row:
    locationIds.append(each_location[0])

for each_location in locations_row:
    freqListLocation = [0] * len(vocab)
    x = each_location[0].count("_")
    each_location_terms = each_location[x+2::4] #list of all the terms of each image one at a time
    for term in each_location_terms:
        freqListLocation[vocab.index(term)] = each_location[each_location.index(term) + mod]
    freqList.append(freqListLocation)
    
fListOfGivenUser = []

#get the vector of the given location from freqList
for location in locationIds:
    if location == Location:
        fListOfGivenLocation = freqList[locationIds.index(location)]
        break

fVectorOfGivenLocation = np.array(fListOfGivenLocation).astype(np.float)


# # similarity calculation

# In[5]:


from scipy import spatial
similarityList = []
most_sim_locations = []

for eachList in freqList:
    eachVector = np.array(eachList).astype(np.float)
    similarity = 1 - spatial.distance.cosine(fVectorOfGivenLocation, eachVector) #calculate cosine similarity between given location vector and all other
    similarityList.append((similarity, locationIds[freqList.index(eachList)])) #add each similarity value to a list
    
sortedSimilarityList = sorted(similarityList, key = lambda x:x[0], reverse = True) #sort the list in descending order

most_sim_locations = sortedSimilarityList[1:int(k)+1] #get the k most similar locations and their respective scores
print("The "+k+" most similar images and their scores are: ")
for temp in most_sim_locations:
    print(temp[1]+" with score of "+str(temp[0]))


# # Vector normalization

# In[6]:


from sklearn import preprocessing

sim_location = []

#add the given location vector to a frequency list of all locations
fVectorOfGivenLocation_listForm = fVectorOfGivenLocation.tolist()
sim_location.append(fVectorOfGivenLocation_listForm)
for first_location in most_sim_locations:
    sim_location.append(freqList[locationIds.index(first_location[1])]) 

#normalise the vectors using l2 norm)
normalizedSimilarLocations = preprocessing.normalize(sim_location, norm='l2')


# # contributing term calculation

# In[9]:


top3Terms = []

print("The top 3 contributing terms for each match are: \n")
for each_k in range(1,int(k)+1):
    print("For match "+str(each_k)+":")
    diffVector = []
    for j in range(len(vocab)):
        if(normalizedSimilarLocations[0][j]!=0 and normalizedSimilarLocations[each_k][j]!=0):
            diffVector.append([abs(normalizedSimilarLocations[0][j]-normalizedSimilarLocations[each_k][j]),j])
        else:
            diffVector.append([float("inf"),j])
    sortedVector = sorted(diffVector, key=lambda x:x[0])
    for l in range(3):
        print(vocab[sortedVector[l][1]])

