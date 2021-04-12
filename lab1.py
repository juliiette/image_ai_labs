import imutils as imutils
import numpy as np
import cv2

img = cv2.imread('images/image.jpg')
img_grey = cv2.imread('images/image.jpg', 0)
cv2.imshow('Image', img_grey)
cv2.imwrite('image_grey.jpg', img_grey)

(h, w, d) = img.shape
print("width: {}\nheight: {}\ndepth: {}".format(w, h, d))

(B, G, R) = img[100, 50]
print("R: {}\nG: {}\nB: {}".format(R, G, B))

roi = img[150:350, 200:400]
# cv2.imshow("ROI", roi)

# resized = cv2.resize(img, (200, 200))
# cv2.imshow("Resized", resized)

h, w = img.shape[0:2]
h_new = 300
ratio = w / h
w_new = int(h_new * ratio)
resized = cv2.resize(img, (w_new, h_new))
print("Resized: {}".format(resized.shape))
# cv2.imshow("Resized", resized)

resized_new = imutils.resize(img, height=300)
# cv2.imshow("Resized", resized_new)

center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotated", rotated)

rotated_new = imutils.rotate(img, -45)
# cv2.imshow("Rotated", rotated_new)

blurred = cv2.GaussianBlur(img, (11, 11), 0)
cv2.imshow("Blurred", blurred)

resized = imutils.resize(img, width=460)
b_resized = imutils.resize(blurred, width=460)
summing = np.hstack((resized, b_resized))
# cv2.imshow("Images", summing)

output = img.copy()
cv2.rectangle(output, (270, 50), (420, 260), (0, 0, 255), 2)
# cv2.imshow("Rectangle", output)

image = np.zeros((200, 200, 3), np.uint8)
cv2.line(image, (0, 0), (200, 200), (255, 0, 0), 5)
# cv2.imshow("Line", image)

image_points = np.zeros((1000, 1000, 3), np.uint8)
points = np.array([[600, 200], [910, 641], [300, 300], [0, 0]])
cv2.polylines(image_points, np.int32([points]), 1, (255, 255, 255))
# cv2.imshow('Image', image_points)

image_circle = np.zeros((200, 200, 3), np.uint8)
output = image_circle.copy()
cv2.circle(output, (100, 100), 50, (0, 0, 255), 2)
cv2.imshow("Circle", output)

font = cv2.FONT_HERSHEY_SIMPLEX
font1 = cv2.FONT_HERSHEY_COMPLEX
font2 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(img,'OpenCV',(10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_4)
cv2.putText(img,'OpenCV',(10, 300), font1, 4, (255, 255, 255), 2, cv2.LINE_4)
cv2.putText(img,'OpenCV',(10,100), font2, 4, (255, 255, 255), 4, cv2.LINE_4)
cv2.imshow("Text", img)

cv2.waitKey(0)
# cv2.destroyAllWindows()
