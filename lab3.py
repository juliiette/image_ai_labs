import cv2
import numpy as np
import math


def presentation(window_name, frame):
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def black_and_white(image_name):
    frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("gray.png", frame)
    return frame


def gaussian(image_name):
    frame = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    blur_kernel_size = (5, 5)
    frame_blur = cv2.GaussianBlur(frame, blur_kernel_size, 0)
    return frame_blur


def canny_contour(image_name):
    canny_low_threshold = 50
    canny_high_threshold = 100
    frame = gaussian(image_name)

    frame_canny = cv2.Canny(frame, canny_low_threshold, canny_high_threshold)
    return frame_canny


def concrete_contour(image_name):
    frame = canny_contour(image_name)
    h, w, x, y = 504, 1532, 341, 778

    area = frame[x:x+h, y:y+w]
    black_img = np.zeros_like(frame)
    black_img[x:x+h, y:y+w] = area
    return area


def hough_lines(image_name):
    image = cv2.imread(image_name)
    frame = canny_contour(image_name)
    cdst = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(frame, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a, b = math.cos(theta), math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 225), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(frame, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 225), 3, cv2.LINE_AA)

    resulted = cv2.addWeighted(image, 0.8, cdstP, 1, 0)
    presentation("Image", resulted)


def process_image(image_name):
    gray_image = cv2.cvtColor(image_name, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 2)
    frame_canny = cv2.Canny(blurred_image, 50, 100)
    lines = cv2.HoughLinesP(frame_canny, 1, np.pi/180, 150, None, 50, 10)
    lines = [] if lines is None else lines

    for (x1, y1, x2, y2) in map(lambda x: x[0], lines):
        if y1 < 450 or y2 < 450:
            continue
        if x1 != x2 and abs((y2-y1)/(x2-x1)) < 0.2:
            continue
        cv2.line(image_name, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)

    cv2.putText(image_name, "Stanko", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255))


def process_video(video_name):
    video_capture = cv2.VideoCapture(video_name)
    frames_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'У відео {frames_count} кадрів')

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            process_image(frame)
            presentation("Video", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            else:
                break

    video_capture.release()


def save_processed(video_name):
    video_capture = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('images/output.mp4', fourcc, 20.0, (1280, 720))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            process_image(frame)
            out.write(frame)
        else:
            break

        video_capture.release()
        out.release()
        cv2.destroyAllWindows()


save_processed("images/road.mp4")
