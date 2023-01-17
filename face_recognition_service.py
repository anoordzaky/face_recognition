import cv2
import numpy as np
import tensorflow as tf
import torch
import logging

from utils import suppress_stdout

with suppress_stdout():
    from face_detection import RetinaFace
    from keras_vggface.vggface import VGGFace
    from keras_vggface.utils import preprocess_input

logger = logging.getLogger(__name__)


class FaceRecognition:
    def __init__(self):

        print("Loading VGGFace Model..")
        with suppress_stdout():
            self.VGGFace = VGGFace(model='resnet50',
                                   include_top=False,
                                   input_shape=(224, 224, 3),
                                   pooling='avg'
                                   )

        print("Loading RetinaFace Model..")

        # use CUDA whenever available, else use CPU
        self.Retina = RetinaFace(0 if torch.cuda.is_available() else -1)

        # initialize Tensorflow graph
        self.graph = tf.get_default_graph()

    def detect_face(self, img):

        # run forward pass to the model
        pred = self.Retina(img)
        # append predictions with confidence higher than 0.85
        face = [pred[i] for i in range(len(pred)) if pred[i][2] > 0.85]

        return face

    def get_embeddings(self, files):

        files = np.array(files, dtype=np.float32)

        # preprocessing for the input of VGGFace
        files = cv2.resize(files, (224, 224), interpolation=cv2.INTER_AREA)
        files = np.expand_dims(files, axis=0)

        inputs = preprocess_input(files, version=2)

        # run the prediction
        with self.graph.as_default():
            embeddings = self.VGGFace.predict(inputs)

        return embeddings
