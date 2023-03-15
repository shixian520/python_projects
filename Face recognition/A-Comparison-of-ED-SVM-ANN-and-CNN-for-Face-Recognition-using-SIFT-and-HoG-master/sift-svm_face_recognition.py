import os,sys
from time import time
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split



# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(data_home = 'D:\\dataset', min_faces_per_person=70, color = True)

n_samples, h, w, _ = lfw_people.images.shape

sift = cv2.xfeatures2d.SIFT_create()

#extracting SIFT features from images
X = []
for i in range(n_samples):
    
    face_image = lfw_people.images[i].astype(np.uint8)
    # face_image = cv2.resize(face_image, (224, 224)).astype(np.uint8)
    
    kp1, des1 = sift.detectAndCompute(face_image,None)
    # print des1.shape
    kmeans = KMeans(n_clusters=1, random_state=0).fit(des1)
    X.append(kmeans.cluster_centers_)
    # X.append(face_image)

X = np.reshape(X,(n_samples,128))

# X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)



# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
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


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

score = clf.score(X_test, y_test)
print('Score:', score)



import pickle
Pkl_Filename = "sift-svm-face-{0:.4f}.pkl".format(score)

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(clf, file)

