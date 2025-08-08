# Vehicle Counting and Tracking
This project implements a real-time vehicle counting and tracking system using YOLOv8 for object detection and the SORT algorithm for multi-object tracking. The system processes video streams, identifies vehicles, tracks their movement, and counts them as they cross a predefined line.

Features
Real-time Object Detection: Utilizes the YOLOv8 model to detect various types of vehicles (cars, trucks, buses, motorbikes).

Multi-Object Tracking: Employs the SORT algorithm to maintain consistent IDs for detected vehicles across frames.

Region of Interest (ROI) Masking: Applies a mask to focus detection on a specific area of the video frame, reducing noise and improving accuracy.

Vehicle Counting: Counts vehicles as they cross a designated virtual line.

Visual Feedback: Displays bounding boxes, tracking IDs, and the total count on the video feed.

Installation
Clone the repository:

git clone <repository_url>
cd <repository_name>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install the required libraries:

pip install numpy opencv-python ultralytics cvzone mpmath

Note: The sort library is typically a custom implementation (e.g., sort.py) and mpmath might not be directly used for limit in this context, but it's listed in the provided code.

Download YOLOv8 weights:
Ensure you have the yolov8l.pt weights file in a directory named Yolo-Weights at the root of your project. You can download it from the Ultralytics YOLOv8 GitHub releases.

./Yolo-Weights/yolov8l.pt

📂 Project Structure
.
├── car-counter.py          # Main script for vehicle detection and counting
├── car2.mp4                # Sample video for testing
├── mask1 (1).png           # Mask image for defining the region of interest
├── sort.py                 # Implementation of the SORT tracking algorithm
└── Yolo-Weights/
    └── yolov8l.pt          # Pre-trained YOLOv8 large model weights

Usage
To run the vehicle counting system, execute the car-counter.py script:

python car-counter.py

The script will open a window displaying the video feed with detected vehicles, their tracking IDs, and the total count.

Configuration:
Video Source:

To use a webcam, uncomment cap = cv2.VideoCapture(0) and comment out cap = cv2.VideoCapture("../video/car2.mp4").

Adjust cap.set(3, 1280) and cap.set(4, 720) for webcam resolution if needed.

Detection Classes: The classNames list defines all detectable objects. The current script filters for "car", "truck", "bus", and "motorbike".

Confidence Threshold: The conf > 0.3 condition filters detections with a confidence score above 30%.

Tracking Line: The limit variable [420, 500, 1200, 500] defines the coordinates of the counting line (x1, y1, x2, y2). Adjust these values to match your video's perspective.

Mask Image: The mask1 (1).png file is used to define the region of interest. Ensure its dimensions are appropriate for your video or adjust the resizing in the code.

⚙️ How it Works
Video Capture: The script reads frames from a video file (car2.mp4) or a webcam.

Masking: A mask image (mask1 (1).png) is applied to each frame to focus the object detection on a specific area, such as a road.

Object Detection (YOLOv8):

The YOLOv8 model processes the masked image region to detect vehicles.

It outputs bounding box coordinates, confidence scores, and class IDs for each detected object.

Object Tracking (SORT):

The Sort tracker takes the current frame's detections and attempts to associate them with existing tracks from previous frames.

It assigns a unique ID to each tracked object and updates its position.

max_age: Maximum number of frames to keep a track alive without new detections.

min_hits: Minimum number of consecutive detections required to establish a track.

iou_threshold: Intersection Over Union threshold for matching detections to existing tracks.

Vehicle Counting:

A virtual line is drawn on the video frame.

The center point of each tracked vehicle's bounding box is monitored.

When a vehicle's center crosses the predefined line, its unique ID is added to totalCount if it hasn't been counted before, preventing double-counting.

Visualization:

Bounding boxes and unique IDs are drawn around tracked vehicles.

The counting line changes color (from red to green) when a vehicle crosses it.

The total count is displayed on the screen.

🤝 Contributing
Feel free to fork the repository, make improvements, and submit pull requests.

