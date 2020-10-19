import cv2
import logging as log
import datetime as dt
import pickle

webcam = cv2.VideoCapture(0)
log.basicConfig(filename='lbp_webcam_info.log', level=log.INFO)
cascades = cv2.CascadeClassifier(
    "recoginition/lbpcascade_frontalface_improved.xml")

anterior = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"personname": 1}
with open("label.pickle", 'rb') as f:
    og_label = pickle.load(f)
    labels = {v: k for k, v in og_label.items()}


while True:
    if not webcam.isOpened():
        print("Webcam is not found.")
        break
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascades.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]

        # recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            color = (255, 255, 255)
            stroke = 2
            name = labels[id_]
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        cv2.imwrite("image-detected-gray.png", roi_gray)
        cv2.imwrite("image-detected.png", roi_frame)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    cv2.imshow("Faces", frame)
    #print("faces detection count: ", len(faces))

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
