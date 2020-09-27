# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:07:13 2020

@author: prakh

CODE SOURCE: https://github.com/xiaochus/YOLOv3

REFERENCE (for original YOLOv3):

    @article{YOLOv3,  
          title={YOLOv3: An Incremental Improvement},  
          author={J Redmon, A Farhadi },
          year={2018} 

"""

import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
import matplotlib.pyplot as plt

#############################################################################################################################

def process_image(img):
    """Resize, reduce and expand image.
    # Argument:
        img: original image.
    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

#############################################################################################################################

def get_classes(file):
    """Get classes name.
    # Argument:
        file: classes name for database.
    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

#############################################################################################################################

def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.
    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()
    
#############################################################################################################################    
    
def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

#############################################################################################################################   

def detect_video(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    video_path = os.path.join("videos", "test", video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    
############################################################################################################################# 
    
#just remove the current version of tensorflow (using command: conda remove tensorflow),
#then install 1.15.0 using command: conda install tensorflow=1.15.0. it works
"""   
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.__version__
"""
yolo = YOLO(0.6, 0.5)
file='D://Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/06-YOLOv3/data/coco_classes.txt'
all_classes = get_classes(file)

#Detecting Images
image_name='SinglePersonSingleCar.jpg'
image_name='SingleCar.jpg'
path='D://Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/06-YOLOv3/data/'+image_name
image = cv2.imread(path)
plt.imshow(image)
image = detect_image(image, yolo, all_classes)
cv2.imwrite('D://Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/06-YOLOv3/data/result_'+image_name, image)


#Detecting Image
# # detect videos one at a time in videos/test folder    
# video = 'library1.mp4'
# detect_video(video, yolo, all_classes)

