import cv2
import numpy as np
import os

IMAGE_FOLDER = "images/image_qualification_01"

success = 0
fail = 0
total = 0

cv2.namedWindow("Gate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gate Detection",900,500)

cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Edges",450,250)


def detect_poles(frame):

    height, width = frame.shape[:2]

    # KEEP YOUR ROI
    roi = frame[int(height*0.01):int(height*0.99), :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray,(3,3),0)

    # vertical gradient
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.uint8(np.absolute(sobelx))

    edges = cv2.Canny(sobelx,25,70)

    # reconnect vertical fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,15))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # remove short vertical noise
    filtered = np.zeros_like(edges)

    min_run = 60
    h, w = edges.shape

    for x in range(w):

        run_start = None

        for y in range(h):

            if edges[y,x] != 0:

                if run_start is None:
                    run_start = y

            else:

                if run_start is not None:

                    if y-run_start >= min_run:
                        filtered[run_start:y,x] = 255

                    run_start = None

        if run_start is not None and h-run_start >= min_run:
            filtered[run_start:h,x] = 255

    cv2.imshow("Edges", cv2.resize(filtered,(450,250)))

    # column strength
    column_strength = np.sum(filtered, axis=0)

    if np.max(column_strength) == 0:
        return []

    # smooth histogram
    column_strength = cv2.GaussianBlur(column_strength.reshape(1,-1),(1,51),0).flatten()

    # get two strongest peaks
    sorted_idx = np.argsort(column_strength)[::-1]

    poles = []
    min_sep = 80

    for idx in sorted_idx:

        if len(poles) == 0:
            poles.append(idx)

        elif abs(idx - poles[0]) > min_sep:
            poles.append(idx)
            break

    return poles


images = sorted(os.listdir(IMAGE_FOLDER))

for img in images:

    if not img.endswith(".png"):
        continue

    total += 1

    path = os.path.join(IMAGE_FOLDER,img)

    frame = cv2.imread(path)

    if frame is None:
        continue

    poles = detect_poles(frame)

    if len(poles) == 2:

        left = min(poles)
        right = max(poles)

        distance = right-left

        h = frame.shape[0]

        cv2.line(frame,(left,0),(left,h),(0,255,0),3)
        cv2.line(frame,(right,0),(right,h),(0,255,0),3)

        cv2.circle(frame,(left,h//2),10,(0,255,0),-1)
        cv2.circle(frame,(right,h//2),10,(0,255,0),-1)

        cv2.putText(frame,
                    f"Distance: {distance}px",
                    (left,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,255,0),
                    3)

        success += 1

    else:

        cv2.putText(frame,
                    "Detection Failed",
                    (50,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    3)

        fail += 1


    display = cv2.resize(frame,(900,500))
    cv2.imshow("Gate Detection", display)

    if cv2.waitKey(1000) == 27:
        break


print("\nDetection Summary")
print("---------------------------")
print("Total Images:", total)
print("Successful:", success)
print("Failed:", fail)

if total > 0:
    print("Success Rate:", round(success/total*100,2), "%")

cv2.destroyAllWindows()
