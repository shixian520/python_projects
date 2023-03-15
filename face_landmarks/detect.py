import os
import sys
import cv2
from imutils import paths
import time
import numpy as np
from tqdm import tqdm

def run_model(orig_image, net):
    img = cv2.resize(orig_image, (112, 112))
    img = img/255.0
    img = img.astype('float32')

    blob = cv2.dnn.blobFromImage(img, size = (112, 112))

    net.setInput(blob)

    result = net.forward()
    return result


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



if __name__ == '__main__':

    landmark_net = cv2.dnn.readNetFromONNX('./landmark.onnx')

    path = './test'

    path = './temp'

    path = 'D:\\Face Work\\spoof\\fail_image\\AsReal'

    imagePaths = sorted(list(paths.list_images(path)))

    start = time.time()

    for imagePath in tqdm(imagePaths):

        # print(imagePath)

        filename = os.path.basename(imagePath)

        # imagePath = './test/crop.png'

        image = cv2.imread(imagePath)

        image = cv2.resize(image, (112, 112))

        # start = time.time()

        landmarks = run_model(image, landmark_net) * 112 * 6

        # # print('Taking time is {}s'.format(time.time() - start))

        # landmarks = landmarks.reshape((68, 2))

        # image = cv2.resize(image, (112 * 6, 112 * 6))

        # font = cv2.FONT_HERSHEY_SIMPLEX

        # for i, landmark in enumerate(landmarks):
        #     # print(landmark)
        #     x = int(landmark[0])
        #     y = int(landmark[1])

        #     # cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
        #     # cv2.putText(image, str(i), (x, y), font, 0.5, (0, 255, 0), 2)
        
        # roll = get_roll(landmarks)
        # # print(roll)
        # if abs(roll) > 30:
        #     if roll > 30:
        #         image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                
        #     else:
        #         image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
            
        # else:
        #     pass
        #     print(roll)
        
        # cv2.imwrite(os.path.join('./result', filename), image)

    print('Taking time is {}s'.format(time.time() - start))




