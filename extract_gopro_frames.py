import cv2
import os

BASE_VIDEO_FOLDER = "gopro videos"

NAV_VIDEO_FOLDER = os.path.join(BASE_VIDEO_FOLDER, "navigation")
QUAL_VIDEO_FOLDER = os.path.join(BASE_VIDEO_FOLDER, "qualification")

NAV_OUTPUT_FOLDER = "images/image_navigation"
QUAL_OUTPUT_FOLDER = "images/image_qualification"

FRAME_SKIP = 10   # ~3 fps if video is 30 fps


def extract_frames(video_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    image_counter = 1

    videos = sorted(os.listdir(video_folder))

    for video in videos:

        if not video.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(video_folder, video)

        print("\nProcessing:", video_path)

        cap = cv2.VideoCapture(video_path)

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:

                filename = f"image_{image_counter}.png"
                save_path = os.path.join(output_folder, filename)

                cv2.imwrite(save_path, frame)

                image_counter += 1

            frame_id += 1

        cap.release()

    print("\nTotal images saved:", image_counter - 1)


print("\n--- Extracting NAVIGATION images ---")
extract_frames(NAV_VIDEO_FOLDER, NAV_OUTPUT_FOLDER)

print("\n--- Extracting QUALIFICATION images ---")
extract_frames(QUAL_VIDEO_FOLDER, QUAL_OUTPUT_FOLDER)

print("\nExtraction complete.")