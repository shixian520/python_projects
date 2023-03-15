import os,sys
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np 
import cv2
from sklearn.cluster import KMeans


male_list = []
male_txt = open('./gender/male.txt', 'r')
lines = male_txt.readlines()
for line in lines:
    line = line.replace('\n', '')
    male_list.append(line)
male_txt.close()

female_list = []
female_txt = open('./gender/female.txt', 'r')
lines = female_txt.readlines()
for line in lines:
    line = line.replace('\n', '')
    female_list.append(line)
female_txt.close()


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(data_home = 'D:\\dataset', min_faces_per_person=10, color = True)

n_samples, h, w, _ = lfw_people.images.shape

X = []
labels = []
sift = cv2.xfeatures2d.SIFT_create()

male_num = 0
X = []
for i in range(n_samples):
    
    face_image = lfw_people.images[i].astype(np.uint8)
    # face_image = cv2.resize(face_image, (224, 224)).astype(np.uint8)
    
    kp1, des1 = sift.detectAndCompute(face_image,None)
    # print des1.shape
    kmeans = KMeans(n_clusters=1, random_state=0).fit(des1)
    X.append(kmeans.cluster_centers_)
    # X.append(face_image)
    name = lfw_people.target_names[lfw_people.target[i]]
    name = name.replace('-', ' ')
    name = name.replace('_', ' ')
    if name in male_list:
        id = 0
        male_num += 1
    elif name in female_list:
        id = 1
    labels.append(id)

X = np.reshape(X,(n_samples,128))

# X = np.reshape(X,(n_samples,128))
labels = np.reshape(labels,(n_samples,))

# X = lfw_people.data

# the label to predict is the id of the person
n_classes = 2

print("n_samples: %d" % n_samples)
print("n_classes: %d" % n_classes)
print("Male: %d" % male_num)
print("Female: %d" % (n_samples - male_num))

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {
    "C": [1e3, 5e3, 1e4, 5e4, 1e5],
    "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



score = clf.score(X_test, y_test)
print('Score:', score)


import pickle

Pkl_Filename = "gender-svm-{0:.4f}.pkl".format(score)
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

