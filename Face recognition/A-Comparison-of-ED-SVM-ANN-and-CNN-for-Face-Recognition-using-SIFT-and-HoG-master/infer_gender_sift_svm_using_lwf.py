import os,sys
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
from imutils import paths
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people

def face_detection(frame):
    # net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
    cols = frame.shape[1]
    rows = frame.shape[0]
    MAX_WIDTH = 1024
    scale = 1   
    if (cols > MAX_WIDTH):
        scale = MAX_WIDTH * 1.0 / cols
        _frame = cv2.resize(frame, (MAX_WIDTH, int(scale * rows)))
    else:
        _frame = frame
    
    rows, cols = _frame.shape[:2]

    rect = []
    for alpha in range(10, 15, 1):
        alpha = alpha/10.0
        for beta in range(0, 11, 10):
            frame = _frame.copy()
            frame = brightness_contrast(frame, alpha, -beta * 10 * (alpha - 1))
            net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confThreshold:
                    x1 = int(detections[0, 0, i, 3] * cols)
                    y1 = int(detections[0, 0, i, 4] * rows)
                    x2 = int(detections[0, 0, i, 5] * cols)
                    y2 = int(detections[0, 0, i, 6] * rows)
                    
                    rect.append([x1,y1,x2,y2])
            if len(rect) != 0:
                return _frame, rect
            
    return _frame, rect

def brightness_contrast(image, alpha, beta):
    rstimg = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return rstimg

def face_recognition_sift(face_image, svm_model):

    face_image = cv2.resize(face_image, (47, 60))

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = sift.detectAndCompute(face_image,None)

    kmeans = KMeans(n_clusters=1, random_state=0).fit(des1)

    sift_feature = kmeans.cluster_centers_

    sift_face_image = cv2.drawKeypoints(gray, kp1, face_image)

    sift_face_image = cv2.resize(sift_face_image, (224, 224))

    name = svm_model.predict(sift_feature)[0]

    return name, sift_face_image

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

if __name__ == '__main__':

    # load face detection model
    modelFile = "model/det_uint8.pb"
    configFile = "model/det.pbtxt"

    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    inWidth = 300
    inHeight = 300
    confThreshold = 0.9

    # load face recognition SVM model
    Pkl_Filename = "gender-svm-0.8603.pkl"

    with open(Pkl_Filename, 'rb') as file:  
        svm_model = pickle.load(file)
    # print(svm_model)


    # load SIFT feature extraction
    sift = cv2.xfeatures2d.SIFT_create()

    # target names  
    target_names = ['Male', 'Female']

    # image path
    path = 'C:\\Users\\Admin\\Downloads\\archive (1) (3)\\lfw_home\\George_W_Bush'

    path = 'C:\\Users\\Admin\\Downloads\\archive (1) (3)\\lfw_home\\lfw_funneled\\Amelia_Vega'

    path = 'C:\\Users\\Admin\\Downloads\\lfw_funneled\\Emily_Robison'

    path = 'C:\\Users\\Admin\\Downloads\\lfw_funneled\\Cyndi_Thompson'

    imagePaths = sorted(list(paths.list_images(path)))

    lfw_people = fetch_lfw_people(data_home = 'D:\\dataset', min_faces_per_person=70, color = True)

    n_samples, h, w, _ = lfw_people.images.shape

    
    num = 0
    for i in range(n_samples):

        # imagePath = 'C:\\Users\\Admin\\Downloads\\archive (1) (3)\\lfw_home\\George_W_Bush\\George_W_Bush_0041.jpg'
        
        face_image = lfw_people.images[i].astype(np.uint8)
        
        name = lfw_people.target_names[lfw_people.target[i]]
        name = name.replace('-', ' ')
        name = name.replace('_', ' ')
        if name in male_list:
            id = 0
        elif name in female_list:
            id = 1
        else:
            id = 2
        

        predict_id, feature_face = face_recognition_sift(face_image, svm_model)


        predict_gender = target_names[predict_id]

        # print(predict_gender)

        if predict_id != id:

            num += 1

            # cv2.imshow('image', face_image)
            # cv2.imshow('SIFT features', feature_face)
            # cv2.waitKey(0)
            
    print(num/n_samples*100)
