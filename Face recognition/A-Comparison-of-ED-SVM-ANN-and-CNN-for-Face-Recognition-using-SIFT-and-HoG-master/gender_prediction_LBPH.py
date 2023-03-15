import os,sys
from time import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

import os
import cv2
import numpy as np 


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

lfw_people = fetch_lfw_people(data_home = 'D:\\dataset', min_faces_per_person=5)

n_samples, h, w = lfw_people.images.shape


X = []
labels = []
for i in range(n_samples):

    face_image = lfw_people.images[i]
    name = lfw_people.target_names[lfw_people.target[i]]

    name = name.replace('-', ' ')
    name = name.replace('_', ' ')

    if name in male_list:
        id = 0
    elif name in female_list:
        id = 1
    X.append(face_image)
    labels.append(id)

# X = np.reshape(X,(n_samples,128))
labels = np.reshape(labels,(n_samples,))

# X = lfw_people.data
# n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names

print("n_samples: %d" % n_samples)
print("n_classes: %d" % 2)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

#This will create a file named trainingData.yml in the model folder, which is trained on the images from dataset folder.

recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists('./model'):
    os.makedirs('./model')

recognizer.train(X_train,y_train)


true = 0
false = 0
for image, id in zip(X_test, y_test):
    predict_id,conf = recognizer.predict(image)
    if conf < 50:
        false += 1
    else:
        if id == predict_id:
            true += 1
        else:
            false += 1

acc = true / (true + false)
print('The acc of model is about ', acc)

recognizer.save('model/LBPH-gender-{0:.4f}.yml'.format(acc))


