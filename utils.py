import io
import sys
import os

import cv2
import numpy as np
from PIL import Image
from scipy import spatial
from contextlib import contextmanager


# Initialize Viola-Jones Algorithm
CASCADE_XML = 'haarcascade_frontalface_default.xml'
vjones = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_XML)

"""
Awalnya saya ingin menggunakan haar cascading supaya proses lebih cepat,
namun ternyata model kurang robust sehingga tidak jadi saya gunakan.
Beberapa fungsi untuk mendukung haar cascading masih ada disini untuk debugging.
"""

##################################
### Image Processing Functions ###
##################################

def convert_image(image):

    # image = cv2.imread(image)
    # resizing to make viola-jones detection time uniform
    image_recolored = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_recolored, (640, 320), interpolation=cv2.INTER_AREA)
    # converting to grayscale to be processed by viola-jones
    image_grayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    return image_resized, image_grayscale


def detect_face(image):
    # this function returns bounding boxes from the detection algorithm
    bbox = vjones.detectMultiScale(image, 1.3, 1)
    return bbox


def crop_face(image, bbox):
    # crops the image based on the detected bounding box from viola-jones
    return image[bbox[0][1]:bbox[0][1]+bbox[0][3], bbox[0][0]:bbox[0][0]+bbox[0][2]]

def crop_face_retina(image, pred):
    #crops image from retinanet output
    bbox = pred[0][0]
    bbox_int = [int(i) for i in bbox]
    return image[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]


def img_cosine_similarity(image1, image2):
    # return the cosine similarity between the images
    return -1*(spatial.distance.cosine(image1, image2) - 1)


# for debugging
def export_face(image):
    image = convert_image(image)
    bbox = detect_face(image)
    return crop_face(image, bbox)
    

##############################
### File Loading Functions ###
##############################

# converts the image from bytes to PIL image
def load_from_bytes(bytes):
    return np.array(Image.open(io.BytesIO(bytes)))


# converts the image from array to bytes
def convert_to_bytes(image):
    assert type(image) == np.ndarray, "Input must be an array."

    bytes = io.BytesIO()
    PIL_image = Image.fromarray(image)

    PIL_image.save(bytes, format='JPEG')

    return bytes

###############################
### Miscellaneous Functions ###
###############################

#supress redundant terminal outputs for better readability
@contextmanager
def suppress_stdout():
    with open(os.devnull,"w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            yield
        finally:
            sys.stdout = old_stdout

