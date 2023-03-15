import os
import sys
import cv2
from imutils import paths
import time
import numpy as np
from tqdm import tqdm
import math
import shutil

PI = 3.1415926

threshold = 40

def run_model(orig_image, net):
    img = cv2.resize(orig_image, (112, 112))
    img = img/255.0
    img = img.astype('float32')

    blob = cv2.dnn.blobFromImage(img, size = (112, 112))

    net.setInput(blob)

    result = net.forward()
    return result

def point2point(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    sum = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return math.sqrt(sum)

def point2line(point,line):
    x1 = line[0]  
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1) 
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def get_roll(landmarks):
    first = 2
    second = 14
    PI = 3.1415926

    x = abs(landmarks[first][0] - landmarks[second][0])
    y = abs(landmarks[first][1] - landmarks[second][1])

    roll = np.arctan(y / x) * 180 / PI
    if (landmarks[first][1] > landmarks[second][1]):
        return -roll
    else:
        return roll

def get_yaw(landmarks):
    
    dist_a = point2point(landmarks[17], landmarks[21])
    dist_b = point2point(landmarks[22], landmarks[26])

    if (dist_a == dist_b):
        return 0
    
    elif (dist_a < dist_b):
        return np.arcsin(1- dist_a / dist_b) * 180 / PI
    
    else:
        return np.arcsin(1- dist_b / dist_a) * 180 / PI


if __name__ == '__main__':

    landmark_net = cv2.dnn.readNetFromONNX('./landmark.onnx')

    path_list = ['C:\\Users\\Honeyman\\Downloads\\landmarks_get_odd\\origin_real']

    imagePaths = []
    for path in path_list:
        imagePaths += sorted(list(paths.list_images(path)))


    # path = 'D:\\Face Work\\spoof\\fail_image\\AsReal\\research'

    # imagePaths = sorted(list(paths.list_images(path)))

    save_path = './origin'

    start = time.time()

    for imagePath in tqdm(imagePaths):

        is_odd = False

        # imagePath = 'D:\\Face Work\\spoof\\fail_image\\AsReal\\research\\new_liveness_406073811.webm_240.png'

        # print(imagePath)

        filename = os.path.basename(imagePath)



        image = cv2.imread(imagePath)

        image = cv2.resize(image, (112, 112))

        landmarks = run_model(image, landmark_net) * 112 * 6

        landmarks = landmarks.reshape((68, 2))

        image = cv2.resize(image, (112 * 6, 112 * 6))

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, landmark in enumerate(landmarks):
            # print(landmark)
            x = int(landmark[0])
            y = int(landmark[1])

            cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
            cv2.putText(image, str(i), (x, y), font, 0.5, (0, 255, 0), 2)
        

        cv2.imwrite(os.path.join('./result', filename), image)

        roll = get_roll(landmarks)
        yaw = get_yaw(landmarks)
        # print(roll)
        if abs(roll) > threshold or abs(yaw) > threshold:
            cv2.imwrite(os.path.join('./result', filename), image)
            is_odd = True


        x_points = [point[0] for point in landmarks]
        y_points = [point[1] for point in landmarks]

        x_len = max(x_points) - min(x_points)
        y_len = max(y_points) - min(y_points)
        ratio = x_len * y_len / ((112 * 6) ** 2)
        # print(ratio)

        if ratio < 0.41:
            is_odd = True

        
        if is_odd:
            # shutil.copyfile(imagePath, os.path.join(save_path, filename))
            shutil.move(imagePath, os.path.join(save_path, filename))

        
        
            
    print('Taking time is {}s'.format(time.time() - start))




