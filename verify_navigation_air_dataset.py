import cv2
import numpy as np
import os

DATASET_PATH = "air_dataset/navigation"
FAILED_PATH = "air_dataset/navigation_failed"

os.makedirs(FAILED_PATH, exist_ok=True)

MIN_POLE_HEIGHT = 100


def detect_vertical_lines(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
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
        threshold=60,
        minLineLength=100,
        maxLineGap=50
    )

    candidates = []

    if lines is None:
        return candidates

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy > dx * 3 and dy > MIN_POLE_HEIGHT:
            candidates.append((x1, y1, x2, y2))

    return candidates


def line_center(line):

    x1,y1,x2,y2=line

    cx=int((x1+x2)/2)

    top=min(y1,y2)
    bottom=max(y1,y2)

    return cx,top,bottom


def choose_best_pair(lines, frame_width):

    best_score = -1
    best_pair = None

    img_center = frame_width / 2

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):

            cx1, top1, bot1 = line_center(lines[i])
            cx2, top2, bot2 = line_center(lines[j])

            if cx2 <= cx1:
                continue

            height1 = bot1 - top1
            height2 = bot2 - top2

            if height1 < MIN_POLE_HEIGHT or height2 < MIN_POLE_HEIGHT:
                continue

            overlap = min(bot1, bot2) - max(top1, top2)

            if overlap <= 0:
                continue

            overlap_ratio = overlap / min(height1, height2)

            if overlap_ratio < 0.5:
                continue

            span = cx2 - cx1

            pair_center = (cx1 + cx2) / 2
            center_error = abs(pair_center - img_center)

            score = span * 0.5 + min(height1, height2) * 0.3 + (300 - center_error) * 0.2

            if score > best_score:
                best_score = score
                best_pair = ((cx1, top1, bot1), (cx2, top2, bot2))

    return best_pair


def process_image(frame):

    lines = detect_vertical_lines(frame)

    if len(lines) < 2:
        return False, frame

    pair = choose_best_pair(lines, frame.shape[1])

    if pair is None:
        return False, frame

    left, right = pair

    lx, ltop, lbot = left
    rx, rtop, rbot = right

    # ---- Bounding Box ----
    top = min(ltop, rtop)
    bottom = max(lbot, rbot)

    x1 = lx
    y1 = top

    x2 = rx
    y2 = bottom

    # draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

    # label
    cv2.putText(
        frame,
        "Navigation Gate",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    # center point
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    cv2.circle(frame, (center_x, center_y), 8, (0,255,255), -1)

    return True, frame


images = sorted(os.listdir(DATASET_PATH))

success = 0
fail = 0

for img in images:

    if not img.endswith(".png"):
        continue

    path = os.path.join(DATASET_PATH, img)

    frame = cv2.imread(path)

    frame = cv2.resize(frame, (960,540))

    ok, result = process_image(frame)

    if ok:
        success += 1
    else:
        fail += 1
        cv2.imwrite(os.path.join(FAILED_PATH, img), result)

    cv2.imshow("Detection", result)

    if cv2.waitKey(5) == 27:
        break


print("\nNavigation Dataset Results")
print("Success:", success)
print("Fail:", fail)

cv2.destroyAllWindows()