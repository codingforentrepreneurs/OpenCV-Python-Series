# joincfe.com/github/
# lessons/
# https://www.youtube.com/playlist?list=PLEsfXFp6DpzRyxnU-vfs3vk-61Wpt7bOS
import datetime
import time
import os
import numpy as np
import cv2
import glob

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)
save_path = 'saved-media/timelapse.mp4'
frames_per_seconds = 24.0
config = CFEVideoConf(cap, filepath=save_path, res='720p')
out = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
seconds_duration = 50
seconds_between_shots = .25
timelapse_img_dir = "images/timelapse/"

if not os.path.exists(timelapse_img_dir):
    os.mkdir(timelapse_img_dir)


now = datetime.datetime.now()
finish_time = now + datetime.timedelta(seconds=seconds_duration)

i = 0

while datetime.datetime.now() < finish_time:
    # Capture frame-by-frame
    ret, frame = cap.read()
    filename = f"{timelapse_img_dir}/{i}.jpg"
    i += 1
    cv2.imwrite(filename, frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # out.write(frame)
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    time.sleep(seconds_between_shots)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

clear_images = True

def images_to_video(out, img_dir, clear_images=True):
    image_list = glob.glob(f"{img_dir}/*.jpg")
    sorted_images = sorted(image_list, key=os.path.getmtime)
    for file in sorted_images:
        image_frame = cv2.imread(file)
        out.write(image_frame)
    if clear_images:
        for file in image_list:
            os.remove(file)

images_to_video(out, timelapse_img_dir, clear_images=False)
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()