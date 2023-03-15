import cv2
from matplotlib.text import get_rotation
from scipy.stats.stats import tiecorrect
import tensorflow as tf
import numpy as np
import time
import onnx
from caffe2.python.onnx import backend
import onnxruntime as ort
import utils.box_utils_numpy as box_utils
import efficientnet.keras as efn 
import efficientnet.tfkeras
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

def run_onnx_model(orig_image, net):
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


class FaceAnalysis():
    def __init__(self):
        self.image_path = None
        self.img = None
        self.stop = False
        self.photocheck_rst = None
        self.photocheck_img = None
        
        self.onnx_path = "models/face.onnx"
        self.predictor = onnx.load(self.onnx_path)
        onnx.checker.check_model(self.predictor)
        onnx.helper.printable_graph(self.predictor.graph)
        self.predictor = backend.prepare(self.predictor, device="CPU")  # default CPU

        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.confThreshold = 0.7

        self.img_size = 224
        self.photocheckModel_path = './models/cover_model.h5'
        self.photocheckModel = self.build_model()
        self.photocheckModel.load_weights(self.photocheckModel_path)

        self.landmark_model_path = './models/landmarks.onnx'
        self.landmark_model = cv2.dnn.readNetFromONNX(self.landmark_model_path)

    def build_model(self):
        input_size = (self.img_size, self.img_size, 3)
        base_model = efn.EfficientNetB4(weights=None,include_top=False, input_shape=input_size)
        x = base_model.output
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dense(1024, activation = 'relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation = 'relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation = 'relu', name='fc3')(x)
        x = Dropout(0.5)(x)
        output = Dense(2, activation='softmax')(x)
        model = Model(inputs=[base_model.input], outputs=[output])
        
        return model

    def brightness_contrast(self, image, alpha, beta):
        img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return img

    def face_detection(self, frame):
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
                _frame = self.brightness_contrast(_frame, alpha, -beta * 10 * (alpha - 1))
                self.face_net.setInput(cv2.dnn.blobFromImage(
                    _frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
                detections = self.face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.confThreshold:
                        x1 = int(detections[0, 0, i, 3] * cols)
                        y1 = int(detections[0, 0, i, 4] * rows)
                        x2 = int(detections[0, 0, i, 5] * cols)
                        y2 = int(detections[0, 0, i, 6] * rows)

                        rect.append([x1, y1, x2, y2])
                if len(rect) != 0:
                    return _frame, rect

        return _frame, rect

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def get_rotate(self, face_image):

        face_image = cv2.resize(face_image, (112, 112))
        landmarks = run_onnx_model(face_image, self.landmark_model)
        landmarks = landmarks.reshape((68, 2))
        roll = get_roll(landmarks)

        return roll


    def face_detect(self, image):
        orig_img = image.copy()
        cols = orig_img.shape[1]
        rows = orig_img.shape[0]
        MAX_WIDTH = 1024
        scale = 1
        if (cols > MAX_WIDTH):
            scale = MAX_WIDTH * 1.0 / cols
            orig_img = cv2.resize(orig_img, (MAX_WIDTH, int(scale * rows)))

        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, _, _ = self.predict(orig_img.shape[1], orig_img.shape[0], confidences, boxes, self.confThreshold)
        rect = []
        square = 0
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            box_square = (box[2] - box[0]) * (box[3] - box[1])
            if square < box_square:
                square = box_square
                rect = box

        if len(rect) != 0:
            return orig_img, rect
        else:
            return orig_img, rect
    
    def get_cover_result(self, face_image):
        face_image = face_image/255.0
        face_image = face_image.astype(dtype="float32")
        face_image = np.expand_dims(face_image, 0)
        output_data = self.photocheckModel.predict(face_image)
        index = np.argmax(output_data)
        return index

    def detectCover(self, image):
        if image is not None:
            self.img = image
        else:
            self.img = None
            print('Image Error')
            return "Fail", image

        isPhotocover = False
        real_face_count = 0
        
        # resized_img, face_rect = self.face_detection(image)
        resized_img, face_rect = self.face_detect(image)
        if len(face_rect) != 0:
            x1 = face_rect[0]
            y1 = face_rect[1]
            x2 = face_rect[2]
            y2 = face_rect[3]

            orig_face = resized_img[y1:y2, x1:x2]

            rows, cols = resized_img.shape[:2]
            if x1 >= cols or y1 >= rows:
                return "Fail", image
            width = x2 - x1
            height = y2 - y1
            start_x = x1 - int(width * 0.7)
            if start_x < 0:
                start_x = 0
            start_y = y1 - int(height * 0.5)
            if start_y < 0:
                start_y = 0
            end_x = start_x + int(width*2.4)
            if end_x > cols:
                end_x = cols
            end_y = start_y + int(height * 2.2)
            if end_y > rows:
                end_y = rows

            image = resized_img[start_y:end_y, start_x:end_x]

            rows, cols = image.shape[:2]
            if cols == 0 or rows == 0:
                return "Fail", image
            real_face_count += 1
            if rows != self.img_size or cols != self.img_size:
                image = cv2.resize(
                    image, (self.img_size, self.img_size))


            idx = self.get_cover_result(image)
            if idx != 0:
                self.rotate = self.get_rotate(orig_face)
                if abs(self.rotate) > 30:
                    print('Rotate face 90 and repredict...')
                    if self.rotate > 30:
                        image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                else:
                    print('Rotate face 180 and repredict...')
                    image = cv2.rotate(image, cv2.ROTATE_180)
                
                idx = self.get_cover_result(image)
                if idx != 0:
                    print('Photocover Face')
                    isPhotocover = True
                    cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 0, 255), thickness = 3)
        else:
            print('No Face')

        if not isPhotocover:
            return 'Real', resized_img
        else:
            return 'PhotoCover', resized_img

    

    def getResult(self, image, photocheck_rst, photocheck_img):
        print("Processing Photocheck Engine... ...")
        start = time.time()

        self.overall = True
        
        result, result_image = self.detectCover(image)

        print("Photocover time is ", time.time() - start, "s")

        if result == "Real":
            photocheck_rst.append("Pass")
            photocheck_img.append(None)
            return photocheck_img, photocheck_rst
        else:
            photocheck_rst.append("Fail")
            photocheck_img.append(result_image)
            return photocheck_img, photocheck_rst
