import cv2
import os

cap = cv2.VideoCapture(0)
#update folder location for each user
directory = "images/this.guy/"

i = 0;
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if i % 15 == 0:
        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(os.path.join(directory, 'this.guy'+str(i)+'.png'), frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    i+=1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()