
import os,sys
from time import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import cv2
import numpy as np 



# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(data_home = 'D:\\dataset', min_faces_per_person=70)

print(lfw_people.target_names)

n_samples, h, w = lfw_people.images.shape



# Preparing dataset
X = []
for i in range(n_samples):
    face_image = lfw_people.images[i].astype(np.uint8)
    X.append(face_image)


# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("n_samples: %d" % n_samples)
# print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#This will create a file named trainingData.yml in the model folder, which is trained on the images from dataset folder.

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

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

recognizer.save('model/LBPH-face-{0:.4f}.yml'.format(acc))

