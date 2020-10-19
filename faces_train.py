import os
import cv2
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "images")

cascades = cv2.CascadeClassifier(
    "recoginition/lbpcascade_frontalface_improved.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


x_train = []
y_labels = []

current_id = 0
label_ids = {}

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if(file.endswith("png") or file.endswith("jpg")):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            #print("labelids = ", label_ids)
            # x_train.append(path)
            # y_labels.append(label)

            pil_image = Image.open(path).convert("L")  # gray scale
            image_array = np.array(pil_image, "uint8")
            # print(image_array)

            faces = cascades.detectMultiScale(
                image_array, scaleFactor=1.1, minNeighbors=5)

            for x, y, w, h in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("label.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

cv2.destroyAllWindows()
