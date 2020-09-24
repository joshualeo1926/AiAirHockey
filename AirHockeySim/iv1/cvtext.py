import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 60)

with open('.\\calib_save.txt', 'r') as c:
    lines = c.readlines()

for i in range(len(lines)):
    lines[i] = float(lines[i].replace('\n', ''))
transformation_matrix = np.array([[lines[0], lines[1], lines[2]],\
            [lines[3], lines[4], lines[5]], [lines[6], lines[7], lines[8]]])

maxWidth = int(lines[9])
maxHeight = int(lines[10])

while(True):
    ret, frame = cap.read()
    frame = cv2.warpPerspective(frame, transformation_matrix, (maxWidth, maxHeight))


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()