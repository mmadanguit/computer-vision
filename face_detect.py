""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((21, 21), 'uint8')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test.webm')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:

        # Blur face
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)

        # Draw smile
        center_sm = (x+(5*w/10),y+(7*h/10))
        axes_sm = (w/5,y/8)
        angle_sm = 0
        startAngle_sm = 20
        endAngle_sm = 160
        color_sm = (0, 0, 0)
        thickness_sm = 5

        frame = cv2.ellipse(frame, center_sm, axes_sm, angle_sm, startAngle_sm, endAngle_sm, color_sm, thickness_sm)

        # Draw left eye
        center_le = (x+(3*w/10),y+(4*h/10))
        radius_le = w/8
        color_le = (255, 255, 255)
        thickness_le = -1

        frame = cv2.circle(frame, center_le, radius_le, color_le, thickness_le)

        # Draw left pupil
        radius_lp = w/16
        color_lp = (0, 0, 0)

        frame = cv2.circle(frame, center_le, radius_lp, color_lp, thickness_le)

        # Draw right eye
        center_re = (x+(7*w/10),y+(4*h/10))

        frame = cv2.circle(frame, center_re, radius_le, color_le, thickness_le)

        # Draw right pupil
        frame = cv2.circle(frame, center_re, radius_lp, color_lp, thickness_le)

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
