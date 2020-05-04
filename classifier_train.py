import numpy as np
import os
import cv2
from PIL import Image

def train_classifier(data_dir):
    # to check the list of Data_dir
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        # Opening image and converting into a gray scale
        img = Image.open(image).convert('L')
        # Converting image into a numpy array
        imageNp =  np.array(img, 'uint8')
        # splitting path from image and index [1] has user name so we will split it
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    # Now converting id list into numpy format
    ids = np.array(ids)

    # Now recognize part
    clf = cv2.face.LBPHFaceRecognizer_create()

    # Now Train the faces and ids
    clf.train(faces, ids)
    clf.write("classifier.yml")

train_classifier("data")