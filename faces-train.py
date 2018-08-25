import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []
files_to_remove = []
dirs_to_remove = []

for root, dirs, files in os.walk(image_dir):
    i = 0
    for file in files:
        if len(files) == 0:
            dirs_to_remove.append(root)
        if file.lower().endswith("png") or file.lower().endswith("jpg") or file.lower().endswith("gif"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")  # grayscale
            size = (320, 180)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            #Histrogram equalization
            equalization = cv2.equalizeHist(image_array)
            #Bilaterally filtered
            filtered = cv2.bilateralFilter(equalization, 9, 10, 10)
            faces = face_cascade.detectMultiScale(filtered, scaleFactor=1.05, minNeighbors=6)

            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    roi = filtered[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    print("adding face for: " + label)
            else:
                files_to_remove.append(os.path.join(root, file))
with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")

#cleanup unusuable images
for fileLoc in files_to_remove:
    print("deleting file: " + fileLoc)
    #os.remove(fileLoc)
for dirsLoc in dirs_to_remove:
    print ("deleting file: " + dirsLoc)
    # os.removedirs(dirsLoc)
print ("removing " + str(len(files_to_remove)) + "files")