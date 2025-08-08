import numpy as np
from mpmath import limit
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../video/car2.mp4")


model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask1 (1).png")
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limit  = [420, 500,1200, 500]
totalCount = []



while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break  # End of video stream

    # Resize the mask to match the dimensions of the video frame
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h))
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                #                    offset=3)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=9)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))



    resultsTracker = tracker.update((detections))
    cv2.line(img,(420, 500),(1200, 500),(0,0,255),5)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9, rt= 2,colorR=(255,0,255))
        # cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
        #                    offset=10)

        center_x, center_y = x1+w//2 , y1+h//2
        cv2.circle(img,(center_x,center_y), 4, (0,255,0),cv2.FILLED)


        if limit[0] < center_x < limit[2] and limit[1] - 10 < center_y < limit[3] +10:
            if totalCount.count(Id)== 0:
                totalCount.append(Id)
                cv2.line(img, (420, 500), (1200, 500), (0, 255, 0), 5)
        cvzone.putTextRect(img, f'Count:{len(totalCount)}', (50, 50), colorR=(0, 0, 0), colorT=(255, 255, 255))

    cv2.imshow("Image", img)
    # cv2.imshow("imgRegion",imgRegion)
    cv2.waitKey(1)