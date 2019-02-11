
# coding: utf-8

# # Mapping values from xml file

# In[54]:


import os 
import numpy as np
from scipy.spatial.distance import cosine as cs
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
# print(len(mapping))


# # Enter Input Values

# In[55]:


LocationID = input("Enter the Location id:")
Location = mapping[LocationID]
print(Location)
model = input("Enter the model: ")
print(model)
k = input("Enter value of k: ")


# # Open and read each location files

# In[56]:


folder = "/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/descvis/img/"
location_names = list(mapping.values())
fileList = []

#fileList contains list of tuples which are of the form [('location file path', 'location') for all other locations other than the input location
x = len(mapping)
for i in range(0,x):
    if i != (int(LocationID)-1):
        fileList.append((folder + location_names[i] + " " + model + ".csv", location_names[i]))

#givenFile contains input location file path
givenFile = folder + mapping[LocationID] + " " + model + ".csv"

locationList = []

#open the Input location file of the given model
with open (givenFile) as f:
    givenLocationData = f.read() #contains all the data within the input location file
    givenLocationList = (givenLocationData.split("\n")) #list of all the images followed by their features wrt that model 
    for each in givenLocationList:
        locationList.append((each.split(","))[:])#list of all lists of all the image feature vectors  


# # calculate similarity based on Euclidean distance

# In[57]:


from scipy import spatial
averageDistance = []
imgximg_sim = {}
for each in fileList:
    eachFile = each[0] #contains location file path
    title = each[1] #contains location title
    imageDistance = []
    imgximg_sim[each] = []
    
    #open each file from the fileList 
    with open (eachFile) as eFile:
        fileData = (eFile.read()).split("\n")[:-1]#read data from each of the file in fileList
    otherLocationList = []

    for eachRow in fileData:
        eachRow = eachRow.split(",")
        otherLocationList.append(eachRow)#append all the image vectors of all the locations other than input location

    for eachGivenImage in locationList:
        eachGivenImageVector = np.array(eachGivenImage[1:]).astype(np.float)#convert list to numpy array
        for eachImage in otherLocationList:
            eachImageVector = np.array(eachImage[1:]).astype(np.float)#convert list to numpy array
            if len(eachImageVector) != 0 and len(eachGivenImageVector) != 0:
                similarity = spatial.distance.euclidean(eachGivenImageVector, eachImageVector)
                similarity = 1 / (1 + similarity)
                imgximg_sim[each].append([(similarity), eachImage[0], eachGivenImage[0]])


# # Sort and print the similarity distances, location names and 3 contributing image pairs

# In[58]:


similarities = []
for otherLocation in imgximg_sim:
    count = 0
    sim = 0
    for img in imgximg_sim[otherLocation]:
        sim+= img[0]
        count+= 1
    similarities.append([otherLocation, sim/count])

similarities = sorted(similarities, key=lambda x:x[1], reverse = True)
# print(similarities)

imageId = []
print("The top "+str(k)+" most similar locations and their scores are: \n")
for i in range(0,int(k)):
    similarLocation = similarities[i][0]
    print(similarLocation[1]+" with score: "+str(similarities[i][1]))
    imagewa = sorted(imgximg_sim[similarLocation], key=lambda x:x[0], reverse = True)
    print("For the similarity between input location and "+similarLocation[1]+" the 3 pairs of images are:")
    for x in imagewa[0:3]:
        print(x[1]+" and "+x[2])
    print("\n")

