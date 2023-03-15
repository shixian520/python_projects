import os
import sys

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import imutils
import numpy as np
import random

def redetect(image):
    try:
        rgb = image.copy()
        small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        temp_img = rgb.copy()

        height, width, _ = rgb.shape

        boxes = []

        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            if w < 5 or h < 5:
                continue
            # cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
            boxes.append([x, y, x + w, y + h])

        boxes = sorted(boxes, key=lambda box: box[0])

        new_x1 = sorted(boxes, key=lambda box: box[0])[0][0]
        new_y1 = sorted(boxes, key=lambda box: box[1])[0][1]
        new_x2 = sorted(boxes, key=lambda box: box[2])[-1][2]
        new_y2 = sorted(boxes, key=lambda box: box[3])[-1][3]

        max_rect = temp_img[max(new_y1 - 1, 1): new_y2, max(new_x1 - 2, 0):new_x2 + 2]

        return max_rect
    
    except:
        return image

# reading male/female name list
male_list = []
male_txt = open('female.txt', 'r')
lines = male_txt.readlines()
for line in lines:
    line = line.replace('\n', '')
    male_list.append(line)
male_txt.close()

for name in male_list:
    if 'Q' in name or 'O' in name:
        pass
    else:
        continue

    print(name)
    # name = "Kelly JONAH KELLY-MEYERS"
    # name = "Nimo Ali Mahamed ROOBLE"
    # name = "QQ man QQ"
    font_type = "DBSILLC.ttf"


    back_color = (178, 176, 163)
    back_color = (226, 209, 181)
    back_color = (199, 197, 184)
    # name = reader.readtext(imagePath)
    # name = name[0][1]
    # draw image from ttf and ocr result

    font = ImageFont.truetype(font_type,40)




    w,h = font.getsize(name)
    spacing = 1
    W = w * 2
    H = h * 2

    img=Image.new("RGBA", (W,H),(back_color))

    draw = ImageDraw.Draw(img)
    # draw.text(((W-w)/2,(H-h)/2), name, font=font, fill=0)

    x = int(W * 0.25)

    for letter in name:

        draw.text((x, int(H * 0.25)),letter,(0,0,0),font=font, spacing = 0.1)
        letter_space = font.getsize(letter)[0]
        x += (letter_space + spacing)

    draw = ImageDraw.Draw(img)
    img.save("temp.png")

    # # get the boundary of the image
    image  = cv2.imread("temp.png")

    image = redetect(image)

    index  = random.choice([0,1,3])

    if index == 0:
        pass
    else:
        image = cv2.medianBlur(image,index)




    save_path = './name/' + 'font_make_' + name + '.jpg'

    cv2.imwrite(save_path, image)

    # cv2.imshow("image", crop_image)
    # cv2.waitKey(0)


    # Get the similarity between the original image and generated image










