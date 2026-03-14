import cv2
import numpy as np
import os

IMAGE_FOLDER = "images"

MIN_POLE_HEIGHT = 80


# ---------------------------------------------------
# COLOR MASK (RED + GREEN)
# ---------------------------------------------------

def get_gate_color_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # red ranges
    lower_red1 = np.array([0,90,40])
    upper_red1 = np.array([15,255,255])

    lower_red2 = np.array([160,90,40])
    upper_red2 = np.array([180,255,255])

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(red1, red2)

    # green range
    lower_green = np.array([45,120,60])
    upper_green = np.array([80,255,255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.bitwise_or(red_mask, green_mask)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    return mask


# ---------------------------------------------------
# DETECT VERTICAL EDGES
# ---------------------------------------------------

def detect_vertical_lines(frame, mask):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel = np.absolute(sobel)

    if np.max(sobel) != 0:
        sobel = np.uint8(255 * sobel / np.max(sobel))
    else:
        sobel = np.uint8(sobel)

    _, edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    # keep only color regions
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=50,
        minLineLength=60,
        maxLineGap=40
    )

    candidates = []

    if lines is None:
        return candidates

    for line in lines:

        x1,y1,x2,y2 = line[0]

        dx = abs(x2-x1)
        dy = abs(y2-y1)

        if dy > dx*3 and dy > MIN_POLE_HEIGHT:
            candidates.append((x1,y1,x2,y2))

    return candidates


# ---------------------------------------------------
# LINE CENTER
# ---------------------------------------------------

def line_center(line):

    x1,y1,x2,y2 = line

    cx = int((x1+x2)/2)

    top = min(y1,y2)
    bottom = max(y1,y2)

    return cx,top,bottom


# ---------------------------------------------------
# CHOOSE BEST POLE PAIR
# ---------------------------------------------------

def choose_best_pair(lines, frame_width):

    best_span = 0
    best_pair = None

    for i in range(len(lines)):
        for j in range(i+1,len(lines)):

            cx1,top1,bot1 = line_center(lines[i])
            cx2,top2,bot2 = line_center(lines[j])

            if cx2 <= cx1:
                continue

            height1 = bot1-top1
            height2 = bot2-top2

            if height1 < MIN_POLE_HEIGHT or height2 < MIN_POLE_HEIGHT:
                continue

            span = cx2 - cx1

            if span > best_span:
                best_span = span
                best_pair = ((cx1,top1,bot1),(cx2,top2,bot2))

    return best_pair


# ---------------------------------------------------
# PROCESS IMAGE
# ---------------------------------------------------

def process_image(path):

    frame = cv2.imread(path)

    if frame is None:
        return

    frame = cv2.resize(frame,(960,540))

    mask = get_gate_color_mask(frame)

    lines = detect_vertical_lines(frame, mask)

    pair = choose_best_pair(lines, frame.shape[1])

    height,width,_ = frame.shape
    image_center = width//2

    cv2.line(frame,(image_center,0),(image_center,height),(255,255,0),2)

    if pair is not None:

        left,right = pair

        lx,ltop,lbot = left
        rx,rtop,rbot = right

        cv2.line(frame,(lx,ltop),(lx,lbot),(0,255,0),3)
        cv2.line(frame,(rx,rtop),(rx,rbot),(0,255,0),3)

        top = min(ltop,rtop)
        bottom = max(lbot,rbot)

        cv2.rectangle(frame,(lx,top),(rx,bottom),(0,255,0),3)

        gate_center_x = int((lx+rx)/2)
        gate_center_y = int((top+bottom)/2)

        cv2.circle(frame,(gate_center_x,gate_center_y),10,(0,0,255),-1)

        error = gate_center_x - image_center

        print(path,"alignment error:",error)

    cv2.imshow("Mask",mask)
    cv2.imshow("Navigation Gate Detection",frame)

    cv2.waitKey(0)


# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------

images = sorted(os.listdir(IMAGE_FOLDER))

for img in images:

    if img.endswith(".png"):

        process_image(os.path.join(IMAGE_FOLDER,img))

cv2.destroyAllWindows()