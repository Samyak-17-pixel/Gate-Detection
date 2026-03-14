import cv2
import os

# -----------------------------
# CAMERA SETTINGS
# -----------------------------

CAMERA_DEVICE = "/dev/video2"   # change if needed

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# -----------------------------
# DATASET FOLDER
# -----------------------------

dataset_folder = "air_dataset/navigation"
os.makedirs(dataset_folder, exist_ok=True)

image_count = len(os.listdir(dataset_folder)) + 1

print("\nNavigation Gate Collector")
print("Press 's' to save image")
print("Press ESC to exit\n")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame capture failed")
        break

    display = frame.copy()

    cv2.putText(display,
                "NAVIGATION DATASET",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Navigation Gate Capture", display)

    key = cv2.waitKey(1) & 0xFF


    if key == ord('s'):

        filename = f"nav_{image_count}.png"
        path = os.path.join(dataset_folder, filename)

        cv2.imwrite(path, frame)

        print("Saved:", filename)

        image_count += 1


    elif key == 27:
        break


cap.release()
cv2.destroyAllWindows()
