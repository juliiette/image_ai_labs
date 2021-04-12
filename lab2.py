import cv2
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def face(image_name):
    scaling_factor = 0.5
    frame = cv2.imread(image_name)
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (173, 20, 204), 3)

    cv2.imshow("Face detection", frame)
    cv2.waitKey(0)
    print(f'Found {len(face_rects)} faces.')


def detail_face(image):
    frame = cv2.imread(image)
    grey_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_filter, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (173, 20, 204), 10)
        roi_grey = grey_filter[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_grey, 1.6, 20)
        eye = eye_cascade.detectMultiScale(roi_grey, 1.1, 20)

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (52, 122, 98), 10)

        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (157, 202, 205), 10)

    cv2.imshow("Detail face", frame)
    cv2.waitKey(0)
    return len(faces)


def live_face():
    cap = cv2.VideoCapture(0)
    while True:
        time.sleep(0.02)
        ret, frame = cap.read()
        detail_face(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def bodies_num(image_name):
    scaling_factor = 0.5
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    image = cv2.imread(image_name)
    image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    people_rects = hog.detectMultiScale(image, winStride=(8, 8), padding=(30, 30), scale=1.06)

    for (x, y, w, h) in people_rects[0]:
        cv2.rectangle(image, (x, y), (x+w, y+h), (173, 20, 204), 2)

    cv2.imshow('Bodies', image)
    cv2.waitKey(0)
    print(f'Found {len(people_rects)} bodies.')


def video_people(video):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.startWindowThread()
    cap = cv2.VideoCapture(video)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 560))
        grey_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])

        for (xa, ya, xb, yb) in boxes:
            cv2.rectangle(frame, (xa, ya), (xb, yb), (173, 20, 204), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(0):
            break

    cap.release()


video_people('images/video.mov')
