import numpy as np
import cv2
import pickle
from collections import Counter

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

font = cv2.FONT_HERSHEY_SIMPLEX
colorWhite = (255, 255, 255)
colorRed = (0, 0, 255)
colorYellow = (0, 200, 200)
colorGreen = (0, 255, 0)
thisColor = colorWhite
stroke = 1
result_array = []
result_dict = {}
result_interval = 15


cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Histrogram equalization
    equalization = cv2.equalizeHist(gray)
    # Bilaterally filtered
    filtered = cv2.bilateralFilter(equalization, 9, 10, 10)
    faces = face_cascade.detectMultiScale(filtered, scaleFactor=1.05, minNeighbors=6)

    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = filtered[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if 60 <= conf <= 100:

            name = labels[id_]
            if name == 'brandon' and conf >= 95:
                thisColor = colorRed
            elif 60 <= conf <= 80:
                thisColor = colorYellow
            elif 80 <= conf <= 99:
                thisColor = colorGreen
            else:
                thisColor = colorWhite

            if result_interval % 10 == 0:
                if name not in result_dict:
                    result_dict.update({name: [int(conf)]})
                else:
                    result_dict[name].append(int(conf))

            stroke = 2
            combined_name = name + " confidence: " + str(int(conf)) +"%"
            cv2.putText(frame, combined_name, (x, y), font, .7, thisColor, stroke, cv2.LINE_AA)

            if name == 'brandon' and conf >= 95:
                thisColor = colorRed
            else:
                thisColor = colorWhite

            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), thisColor, stroke)
        else:
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), colorWhite, stroke)
            cv2.putText(frame, "unknown", (x, y), font, .7, colorWhite, stroke, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    result_interval += 1
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

data = Counter(result_dict)
conf_level_dict = dict(data.most_common(1))

print(conf_level_dict)

#keepers
print(list(conf_level_dict.keys())[0])
print(sum(list(conf_level_dict.values())[0]) / len(list(conf_level_dict.values())[0]))


