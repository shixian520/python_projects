import os
import sys
import cv2
from imutils import paths
import time
import numpy as np
from tqdm import tqdm
import math
import shutil

# PI = 3.1415926

threshold = 30


if __name__ == '__main__':

    landmark_net = cv2.dnn.readNetFromONNX('./landmark.onnx')

    # path_list = ['C:\\Users\\Honeyman\\Downloads\\landmarks_get_odd\\origin_real']

    path_list = ['D:\\Face Work\\spoof\\fail_image\\AsReal\\black']

    imagePaths = []
    for path in path_list:
        imagePaths += sorted(list(paths.list_images(path)))


    # path = 'D:\\Face Work\\spoof\\fail_image\\AsReal\\research'

    # imagePaths = sorted(list(paths.list_images(path)))

    save_path = './origin'

    start = time.time()

    for imagePath in tqdm(imagePaths):

        is_odd = False

        filename = os.path.basename(imagePath)

        image = cv2.imread(imagePath)

        color = np.average(image)

        if color > 15:
            is_odd = True
        else:
            is_odd = False
            pass

        
        if is_odd:
            shutil.copyfile(imagePath, os.path.join(save_path, filename))
            # shutil.move(imagePath, os.path.join(save_path, filename))

        
        
            
    print('Taking time is {}s'.format(time.time() - start))




