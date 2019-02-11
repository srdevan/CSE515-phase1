
# coding: utf-8

# # Create Vocabulary of all terms 

# In[36]:


import numpy as np
import pandas as pd

#open the textDescriptor perUser file
f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerUser.txt', "r")
temp = []
vocab = []
key = 0

for line in f:
    b = line.split()
    temp.append(b[1::4])

for x in temp:
    for y in x:
        vocab.append(y)

vocab = sorted(list(set(vocab)))
# print(vocab)


# # Get the userID, Model and value  of k

# In[35]:


userId = input("Enter the user id:")
model = input("Enter the model: ")
if model == "TF":
    mod = 1
elif model == "DF":
    mod = 2
else:
    mod = 3
k = input("Enter value of k: ")


# # term frequency vector creation

# In[30]:


f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerUser.txt', "r")

users_row = []
userIds = []
user_temp = []
each_user_terms = []
terms = []
tf_values = []

#read file and get the user data in a list 
for line in f:
    b = line.split()
    users_row.append(b[0:])

user_terms = []
user = []
freqList = []

#get the userIds of all the users and store in UserIds list
for each_user in users_row:
    userIds.append(each_user[0])
    
for each_user in users_row:
    freqListUser = [0] * len(vocab)
    each_user_terms = each_user[1::4] #list of all the terms of each user one at a time
    for term in each_user_terms:
        freqListUser[vocab.index(term)] = each_user[each_user.index(term) + mod] #create a vector of TF,DF or TF-IDF values of all terms of each user
    freqList.append(freqListUser) #list of all the user vectors

fListOfGivenUser = []

#get the vector of the given user from freqList
for user in userIds:
    if user == userId:
        fListOfGivenUser = freqList[userIds.index(user)]
        break

fVectorOfGivenUser = np.array(fListOfGivenUser).astype(np.float) #convert list to numpy array


# # similarity calculation 

# In[31]:


from scipy import spatial
similarityList = []
most_sim_users = []

for eachList in freqList:
    eachVector = np.array(eachList).astype(np.float)
    similarity = 1 - spatial.distance.cosine(fVectorOfGivenUser, eachVector) #calculate cosine similarity between given user vector and all other
    similarityList.append((similarity, userIds[freqList.index(eachList)])) #add each similarity value to a list
    
sortedSimilarityList = sorted(similarityList, key = lambda x:x[0], reverse = True) #sort the list in descending order

most_sim_users = sortedSimilarityList[1:int(k)+1] #get the k most similar users and their respective scores
print("The "+k+" most similar users and their scores are: ")
for temp in most_sim_users:
    print(temp[1]+" with score of "+str(temp[0]))
# print(most_sim_users)


# # Vector normalization

# In[32]:


from sklearn import preprocessing

sim_user = []

#add the given user vector to a frequency list of all users 
fVectorOfGivenUser_listForm = fVectorOfGivenUser.tolist()
sim_user.append(fVectorOfGivenUser_listForm)

for first_user in most_sim_users:
    sim_user.append(freqList[userIds.index(first_user[1])]) 
    
#normalise the vectors using l2 norm
normalizedSimilarUsers = preprocessing.normalize(sim_user, norm='l2')


# # contributing term calculation

# In[33]:


top3Terms = []

print("The top 3 contributing terms for each match are: \n")
for each_k in range(1,int(k)+1):
    print("For match "+str(each_k)+":")
    diffVector = []
    for j in range(len(vocab)):
        if(normalizedSimilarUsers[0][j]!=0 and normalizedSimilarUsers[each_k][j]!=0):
            diffVector.append([abs(normalizedSimilarUsers[0][j]-normalizedSimilarUsers[each_k][j]),j])
        else:
            diffVector.append([float("Inf"),j])
    sortedVector = sorted(diffVector, key=lambda x:x[0])

# print(sortedVector)
    for l in range(3):
        print(vocab[sortedVector[l][1]])
    print("\n")

