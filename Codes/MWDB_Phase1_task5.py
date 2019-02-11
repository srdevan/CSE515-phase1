
# coding: utf-8

# # Mapping values from xml file

# In[7]:


import pandas as pd
import numpy as np
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


# # Get the LocationID and value of k

# In[8]:


InputLocationID = input("Enter the Location id:")
Location = mapping[InputLocationID]
print(Location)
k = input("Enter value of k: ")


# # for all locations read each model file then find centroid

# In[9]:


from sklearn.cluster import KMeans
from statistics import mean
from sklearn import preprocessing

folder = "/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/descvis/img/"
location_names = list(mapping.values())
EachLocationModelFile = []
locationData = []
locationDict = {}

model_list = {'1':'CM','2':'CM3x3','3':'CN','4':'CN3x3','5':'CSD','6':'GLRLM','7':'GLRLM3x3','8':'HOG','9':'LBP','10':'LBP3x3'}

#traversing through each location number
x = len(mapping)
# print(x)
for location_number in range(0,x):
    locationVector = []
    #traversing through each model of each location
    for each_model in range(1,11):
        model_from_list = model_list[str(each_model)]
        temp = folder + location_names[location_number] + " " + model_from_list + ".csv"
        with open (temp) as f:
            LocationModelData = f.read()
        LocationModelData = (LocationModelData.split("\n")[:-1])
        locationData = []
        #convert to 2D numpy array
        for i in LocationModelData:
            locationData.append(np.array((i.split(","))[1:], dtype=np.float64))
        Model2DArr = np.array(locationData)
        
        kmeans = KMeans(n_clusters = 1, random_state = 0).fit(Model2DArr) #find out the centroid of each model
        tempvector = kmeans.cluster_centers_
        locationVector.append(tempvector) #append the model vector to a list
    locationDict[location_names[location_number]] = locationVector #add all the model vectors to a dictionary
#     locationDict2[location_names[location_number]] = locationVector2


# # get input vector

# In[10]:


GivenLocationVector = []
GivenLocationVector.append(locationDict[location_names[int(InputLocationID) - 1]])


# # calculate similarity between each model of each loc and take mean of similarity vector to get score

# In[11]:


from scipy import spatial

FinalSimList= {}
AggSimList = []
count = 0

#traversing through each location number
y = len(mapping)
# print(y)
for loc in range(0,y):
    if loc==int(InputLocationID) - 1:
        continue
    SimList = []
    sim_prod = 1
    #traversing through each model of each location
    for mod in range(0,10):
        count += 1
        GivenLocMod = np.array(GivenLocationVector[0][mod]).astype(np.float) #convert to numpy array
        OtherLocMod = np.array(locationDict[location_names[loc]][mod]).astype(np.float) 
        ModelSimilarity = spatial.distance.euclidean(GivenLocMod, OtherLocMod)#calculate euclidean distance between two model vectors
        ModelSimilarity = 1 / (1 + ModelSimilarity)
        SimList.append(ModelSimilarity)
#         sim_prod = sim_prod * ModelSimilarity
    FinalSimList[location_names[loc]] = np.array(SimList)
#     AggSimList.append([location_names[loc],np.mean(SimList),sim_prod**0.1])
    AggSimList.append([location_names[loc],np.mean(SimList)]) #take mean of all the values to find top locations
sortedAggSimList = sorted(AggSimList, key = lambda x: x[1], reverse=True)

print("The top "+str(k)+" most similar locations and their scores are: \n")
for i in range(0,int(k)):
    print(sortedAggSimList[i][0])
    print(sortedAggSimList[i][1])
    print("The model contributions for this match are: ")
    print(FinalSimList[sortedAggSimList[i][0]])

