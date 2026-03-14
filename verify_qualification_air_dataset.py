import cv2
import numpy as np
import os

DATASET_PATH = "air_dataset/qualification"
FAILED_PATH = "air_dataset/qualification_failed"

os.makedirs(FAILED_PATH, exist_ok=True)

MIN_POLE_HEIGHT = 60


def detect_vertical_orange_lines(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 70, 70])
    upper_orange = np.array([30, 255, 255])

    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel = np.absolute(sobel)

    if np.max(sobel) != 0:
        sobel = np.uint8(255 * sobel / np.max(sobel))
    else:
        sobel = np.uint8(sobel)

    _, vertical_edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_and(vertical_edges, orange_mask)

    lines = cv2.HoughLinesP(
        combined,
        1,
        np.pi/180,
        threshold=30,
        minLineLength=40,
        maxLineGap=30
    )

    if lines is None:
        return []

    return [tuple(line[0]) for line in lines]


def detect_horizontal_bar(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.absolute(sobel)

    if np.max(sobel) != 0:
        sobel = np.uint8(255 * sobel / np.max(sobel))
    else:
        sobel = np.uint8(sobel)

    _, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi/180,
        threshold=80,
        minLineLength=200,
        maxLineGap=50
    )

    if lines is None:
        return None

    longest = None
    best_len = 0

    for line in lines:

        x1,y1,x2,y2 = line[0]

        dx = abs(x2-x1)
        dy = abs(y2-y1)

        if dx > dy*3:

            length = dx

            if length > best_len:
                best_len = length
                longest = (x1,y1,x2,y2)

    return longest


def choose_best_pair(lines, frame_width):

    best_span = 0
    best_pair = None

    for i in range(len(lines)):
        for j in range(i+1,len(lines)):

            x1,y1,x2,y2 = lines[i]
            x3,y3,x4,y4 = lines[j]

            cx1 = int((x1+x2)/2)
            cx2 = int((x3+x4)/2)

            if cx2 <= cx1:
                continue

            top1 = min(y1,y2)
            bot1 = max(y1,y2)

            top2 = min(y3,y4)
            bot2 = max(y3,y4)

            height1 = bot1-top1
            height2 = bot2-top2

            if height1 < MIN_POLE_HEIGHT or height2 < MIN_POLE_HEIGHT:
                continue

            span = cx2-cx1

            if span > best_span:
                best_span = span
                best_pair = ((cx1,top1,bot1),(cx2,top2,bot2))

    return best_pair


def detect_gate(frame):

    lines = detect_vertical_orange_lines(frame)

    if len(lines) >= 2:

        pair = choose_best_pair(lines,frame.shape[1])

        if pair is not None:

            left,right = pair

            lx,ltop,lbot = left
            rx,rtop,rbot = right

            top = min(ltop,rtop)
            bottom = max(lbot,rbot)

            return lx,top,rx,bottom

    bar = detect_horizontal_bar(frame)

    if bar is not None:

        x1,y1,x2,y2 = bar

        lx = x1
        rx = x2

        top = y1
        bottom = y1 + 200

        return lx,top,rx,bottom

    return None


def process_image(frame):

    rect = detect_gate(frame)

    if rect is None:
        return False,frame

    x1,y1,x2,y2 = rect

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

    cv2.putText(
        frame,
        "Qualification Gate",
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)

    cv2.circle(frame,(cx,cy),8,(0,255,255),-1)

    return True,frame


images = sorted(os.listdir(DATASET_PATH))

success = 0
fail = 0

for img in images:

    if not img.endswith(".png"):
        continue

    path = os.path.join(DATASET_PATH,img)

    frame = cv2.imread(path)
    frame = cv2.resize(frame,(960,540))

    ok,result = process_image(frame)

    if ok:
        success += 1
    else:
        fail += 1
        cv2.imwrite(os.path.join(FAILED_PATH,img),result)

    cv2.imshow("Qualification Detection",result)

    if cv2.waitKey(5)==27:
        break


print("\nQualification Dataset Results")
print("Success:",success)
print("Fail:",fail)

cv2.destroyAllWindows()