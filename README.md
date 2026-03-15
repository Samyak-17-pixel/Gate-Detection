# SAUVC Gate Detection Pipeline

This repository contains the computer vision pipeline used for detecting navigation and qualification gates for an Autonomous Underwater Vehicle (AUV) developed for the **Singapore Autonomous Underwater Vehicle Challenge (SAUVC)**.

The objective of this project is to enable an AUV to autonomously detect underwater gate structures, determine the position of the vertical poles, compute the horizontal distance between them in the camera frame, and use this information to align with the center of the gate before passing through it.

The detection pipeline is implemented using **classical computer vision techniques with OpenCV** so that it can run efficiently on embedded hardware such as the **Jetson Orin Nano**, which is used as the onboard compute unit of the vehicle.

The repository also includes tools for dataset generation, dataset verification, and extraction of frames from GoPro recordings.

---

# Table of Contents

1. Project Objective  
2. Detection Philosophy  
3. Gate Detection Pipeline  
4. Pipeline Diagram  
5. Repository Structure  
6. Detailed Description of Each File  
7. Dataset Generation  
8. Running the Detection  
9. Dependencies  
10. Example Detection Output  
11. Deployment on the AUV  
12. Future Improvements  

---

# Project Objective

In the SAUVC competition environment, the AUV must perform the following sequence autonomously:

1. Start from a designated starting zone
2. Move forward into the arena
3. Detect the gate structure
4. Align itself with the center of the gate
5. Pass through the gate without touching it

To achieve this, the perception system must reliably detect the **vertical poles of the gate** even in challenging underwater conditions.

The output of the detection algorithm is:

**Distance between the left and right poles in pixels**

This value is later used by the control system to compute alignment error and adjust vehicle motion.

---

# Detection Philosophy

Underwater environments degrade visual data due to several factors:

- Light absorption
- Color attenuation
- Surface reflections
- Suspended particles
- Pool floor reflections
- Reduced contrast

Because of these effects, relying purely on **color detection** is unreliable at larger distances.

However, the **geometric structure of the gate remains consistent**, particularly the two tall vertical poles.

Therefore the detection pipeline focuses primarily on **structural edge detection**, specifically identifying strong vertical structures rather than relying only on color segmentation.

---

# Gate Detection Pipeline

The detection algorithm follows a multi-stage pipeline that progressively isolates vertical structures corresponding to gate poles.

The stages are:

1. Image preprocessing  
2. Contrast enhancement  
3. Noise reduction  
4. Vertical gradient extraction  
5. Edge detection  
6. Vertical edge connection  
7. Vertical structure filtering  
8. Column strength analysis  
9. Pole detection  
10. Distance calculation  

Each stage is explained below.

---

## Step 1 – Image Preprocessing

The input image is converted to grayscale.

Grayscale simplifies the data representation and removes dependence on color, allowing the algorithm to focus on intensity gradients.

---

## Step 2 – Contrast Enhancement (CLAHE)

Underwater images often suffer from low contrast.

The algorithm applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to improve local contrast.

CLAHE enhances local brightness variations and reveals structural details such as faint edges.

---

## Step 3 – Noise Reduction

A Gaussian blur filter is applied to reduce high-frequency noise caused by:

- suspended particles
- sensor noise
- lighting artifacts

This smoothing step improves the stability of edge detection.

---

## Step 4 – Vertical Gradient Extraction

Gate poles are vertical structures.

To highlight vertical structures, the algorithm computes the **horizontal image gradient using the Sobel operator**.

This operation emphasizes vertical edges while suppressing horizontal edges.

---

## Step 5 – Edge Detection

Canny Edge Detection is applied to detect strong edges in the image.

Canny performs several operations:

- gradient thresholding
- non-maximum suppression
- edge tracking

The output is a binary edge map.

---

## Step 6 – Vertical Edge Connection

Underwater edges often appear fragmented due to noise.

Morphological operations are applied to reconnect vertical edge fragments.

A vertical structuring element is used so that primarily vertical structures are reinforced.

---

## Step 7 – Vertical Structure Filtering

Random noise produces small edge fragments.

Gate poles extend across a large vertical portion of the image.

Therefore vertical edge segments shorter than a defined threshold are removed.

This filtering step leaves only large vertical structures.

---

## Step 8 – Column Strength Analysis

The filtered edge map is collapsed vertically.

For each image column the algorithm computes the total edge strength.

Columns containing vertical poles accumulate large edge values.

This produces peaks in the column strength histogram.

---

## Step 9 – Pole Detection

The two strongest peaks in the column histogram correspond to:

- Left pole
- Right pole

These x-coordinates represent the detected pole positions.

---

## Step 10 – Pole Distance Calculation

The horizontal pixel distance between poles is computed:

distance = right_pole_x - left_pole_x

This value represents the apparent gate width in the image.

It can be used to estimate alignment and relative position.

---

## Detection Pipeline Diagram

The gate detection system follows a sequential image processing pipeline. Each stage progressively enhances vertical structural features while suppressing noise and irrelevant edges. The goal is to isolate the two vertical poles of the gate and compute the distance between them.

```
Input Image
    │
Grayscale Conversion
    │
CLAHE Contrast Enhancement
    │
Gaussian Blur
    │
Sobel X Gradient (Vertical Edge Emphasis)
    │
Canny Edge Detection
    │
Vertical Morphological Closing
    │
Vertical Edge Filtering
    │
Column Strength Histogram
    │
Pole Detection
    │
Pole Distance Calculation
```

## Explanation of the Pipeline


Each stage progressively removes noise and highlights the vertical structures that correspond to the gate poles.

---

# Repository Structure

The repository is organized to clearly separate the detection algorithms, dataset generation scripts, verification tools, and video frame extraction utilities.

```
gate_detection/
│
├── navigation_gate_detector.py
├── qualification_gate_detector.py
│
├── collect_navigation_air.py
├── collect_qualification_air.py
│
├── verify_navigation_air_dataset.py
├── verify_qualification_air_dataset.py
│
├── extract_gopro_frames.py
│
├── .gitignore
└── README.md
```

Each file in the repository plays a specific role in building, testing, and validating the gate detection pipeline.

---

# Detailed Description of Each File

## navigation_gate_detector.py

This script performs detection of the **navigation gate** used in the SAUVC navigation task.

The navigation gate consists of vertical poles with alternating colored segments. However, underwater color degradation makes color detection unreliable at longer distances. Therefore the algorithm focuses on detecting **strong vertical structural edges**.

Main responsibilities of this script include:

- Loading images from the dataset
- Running the vertical pole detection pipeline
- Identifying the strongest vertical pole candidates
- Computing the horizontal distance between the poles
- Displaying the detection results on the image
- Printing the pole distance in pixels

This script is intended to eventually run onboard the AUV during real-time mission execution.

---

## qualification_gate_detector.py

This script performs detection of the **qualification gate** used during the SAUVC qualification task.

The qualification gate typically consists of two vertical poles connected by horizontal bars. The algorithm ignores horizontal bars and focuses purely on identifying the vertical poles.

The script performs the following tasks:

- Load images from the qualification dataset
- Run the vertical pole detection pipeline
- Detect left and right vertical poles
- Compute pole-to-pole distance
- Display detection visualization
- Calculate detection statistics

The script also reports:

- Total images processed
- Successful detections
- Failed detections
- Detection success rate

This allows the developer to evaluate the reliability of the algorithm before deploying it on the AUV.

---

## collect_navigation_air.py

This script is used to collect **navigation gate images in air** using a camera connected to the system.

The purpose of this script is to create an initial dataset that can be used for:

- algorithm development
- parameter tuning
- debugging detection logic

Images captured in air help verify the detection pipeline before testing underwater.

---

## collect_qualification_air.py

This script captures **qualification gate images in air**.

It functions similarly to the navigation air dataset collector but is specifically used to generate a dataset for the qualification gate.

These datasets allow developers to:

- verify detection reliability
- test different detection parameters
- observe algorithm behaviour under controlled conditions

---

## verify_navigation_air_dataset.py

This script evaluates the performance of the navigation gate detection algorithm using the collected air dataset.

The script performs the following steps:

1. Loads each image from the dataset
2. Runs the navigation gate detection algorithm
3. Displays detection results
4. Tracks detection success statistics

The script outputs:

- number of successful detections
- number of failed detections
- detection success rate

This allows developers to quantitatively measure algorithm performance.

---

## verify_qualification_air_dataset.py

This script verifies the performance of the qualification gate detection algorithm.

The script:

- loads each image from the qualification dataset
- runs the pole detection pipeline
- displays the detection results
- calculates detection accuracy statistics

This verification stage is important before testing the algorithm underwater.

---

## extract_gopro_frames.py

This script extracts image frames from GoPro videos recorded during pool testing.

GoPro recordings provide valuable underwater data that cannot easily be recreated in air datasets.

The script performs the following tasks:

- scans GoPro video folders
- processes multiple video files
- extracts frames at a chosen frame rate
- saves extracted images to dataset folders

These extracted images are then used for algorithm testing and parameter tuning.

---

# Dataset Handling

Large datasets and videos are **not stored inside the repository** to keep the repository lightweight.

The following directories are excluded using `.gitignore`:

```
air_dataset/
images/
gopro videos/
```

These directories contain:

- air dataset images
- extracted underwater images
- GoPro experiment recordings

Datasets can be stored locally or on external storage systems.

---

# Running the Detection

To run the qualification gate detector:

```
python3 qualification_gate_detector.py
```

To run the navigation gate detector:

```
python3 navigation_gate_detector.py
```



The scripts will:

1. Load dataset images
2. Run the detection pipeline
3. Detect vertical poles
4. Compute pole distance
5. Display the detection results

The detected pole distance can then be used by the AUV control system for alignment.

---

# Dependencies

The project requires the following Python libraries:

- OpenCV
- NumPy

Install dependencies using:

```
pip install opencv-python numpy
```


These libraries provide the image processing tools required for the detection pipeline.

---

# Example Detection Output

When the algorithm successfully detects a gate:

- The left vertical pole is marked
- The right vertical pole is marked
- The distance between the poles is displayed

This information indicates the apparent width of the gate in the camera frame.

This value can be used to determine:

- how centered the vehicle is
- how far the gate is
- whether the vehicle should yaw or sway

---

# Deployment on the AUV

The detection system is designed to run on the **Jetson Orin Nano onboard computer** used in the AUV.

The vision pipeline will be integrated with the vehicle's control system to perform the following actions:

- detect gate poles
- compute alignment error
- adjust yaw angle
- perform lateral sway corrections
- move forward through the gate

This allows the vehicle to autonomously navigate through the gate without human intervention.

---

# Future Improvements

Potential improvements to the system include:

- temporal filtering across consecutive frames
- Kalman filtering for stable pole tracking
- depth estimation using pole geometry
- machine learning based gate detection
- integration with ROS2 for full autonomy pipeline

These improvements will further enhance the robustness and reliability of the perception system.

---

# Authors

Developed as part of the **Team Aritra AUV Project** for the **Singapore Autonomous Underwater Vehicle Challenge (SAUVC)**.
