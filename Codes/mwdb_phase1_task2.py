
# coding: utf-8

# # Create Vocabulary of all terms

# In[32]:


import numpy as np
import pandas as pd

#open the textDescriptor perImage file
f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerImage.txt', "r")
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


# # Get the ImageID, Model and value of k

# In[33]:


imageId = input("Enter the image id:")
model = input("Enter the model: ")
if model == "TF":
    mod = 1
elif model == "DF":
    mod = 2
else:
    mod = 3
k = input("Enter value of k: ")


# # term frequency vector creation

# In[34]:


f = open('/Users/shreyasdevan/Desktop/CSE 515/Phase1/Testdata/devset/desctxt/devset_textTermsPerImage.txt', "r")

images_row = []
imageIds = []
image_temp = []
each_image_terms = []
terms = []
tf_values = []

#read file and get the image data in a list
for line in f:
    b = line.split()
    images_row.append(b[0:])

user_terms = []
user = []
freqList = []

#get the imageIds of all the images and store in ImageIds list
for each_image in images_row:
    imageIds.append(each_image[0])

for each_image in images_row:
    freqListImage = [0] * len(vocab)
    each_image_terms = each_image[1::4]#list of all the terms of each image one at a time
    for term in each_image_terms:
        freqListImage[vocab.index(term)] = each_image[each_image.index(term) + mod]
    freqList.append(freqListImage)

fListOfGivenImage = []

#get the vector of the given image from freqList
for image in imageIds:
    if image == imageId:
        fListOfGivenImage = freqList[imageIds.index(image)]
        break

fVectorOfGivenImage = np.array(fListOfGivenImage).astype(np.float)


# # similarity calculation

# In[35]:


from scipy import spatial
similarityList = []
most_sim_images = []

for index,eachList in enumerate(freqList):
    eachVector = np.array(eachList).astype(np.float)
    similarity = 1 - spatial.distance.cosine(fVectorOfGivenImage, eachVector) #calculate cosine similarity between given image vector and all other
    similarityList.append((similarity, imageIds[index])) #add each similarity value to a list

sortedSimilarityList = sorted(similarityList, key = lambda x:x[0], reverse = True) #sort the list in descending order

most_sim_images = sortedSimilarityList[1:int(k)+1] #get the k most similar users and their respective scores
print("The "+k+" most similar images and their scores are: ")
for temp in most_sim_images:
    print(temp[1]+" with score of "+str(temp[0]))


# # Vector normalization

# In[36]:


from sklearn import preprocessing

sim_image = []

#add the given image vector to a frequency list of all images
fVectorOfGivenImage_listForm = fVectorOfGivenImage.tolist()
sim_image.append(fVectorOfGivenImage_listForm)
for first_image in most_sim_images:
    sim_image.append(freqList[imageIds.index(first_image[1])]) 
    
#normalise the vectors using l2 norm)
normalizedSimilarImages = preprocessing.normalize(sim_image, norm='l2')


# # contributing term calculation

# In[37]:


top3Terms = []

print("The top 3 contributing terms for each match are: \n")
for each_k in range(1,int(k)+1):
    print("For match "+str(each_k)+":")
    diffVector = []
    for j in range(len(vocab)):
        if(normalizedSimilarImages[0][j]!=0 and normalizedSimilarImages[each_k][j]!=0):
            diffVector.append([abs(normalizedSimilarImages[0][j]-normalizedSimilarImages[each_k][j]),j])
        else:
            diffVector.append([float("inf"),j])
    sortedVector = sorted(diffVector, key=lambda x:x[0])
    for l in range(3):
        print(vocab[sortedVector[l][1]])
    print("\n")

