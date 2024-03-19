import math
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

video = cv2.VideoCapture("../Project2- People counter/people.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("../Project2- People counter/mask.png")
# Tracking  the classes
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 550, 735, 489]
model = YOLO('../YOLO-Weights/yolov8l.pt')
totalcountUp = []
totalcountDown = []
while True:
    success, img = video.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("../Project2- People counter/graphics-1.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            CurrentCLASS = classNames[cls]
            if CurrentCLASS == "person" and conf > 0.3:
                #cvzone.putTextRect(img, f"{CurrentCLASS} {conf} ", (max(0, x1), max(35, y1)), scale=1, thickness=0)
                #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 250, 0), 2)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, currentArray])

        resultTracker = tracker.update(detections)
        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 3)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 3)
        for result in resultTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2-x1, y2-y1
            # YOLO DETECTIONS
            #cv2.rectangle(img, (x1, y1), (x2, y2), color=(250, 0, 0), thickness=2)
            #cvzone.cornerRect(img, (x1, y1, w, h), l=9,colorR=(250, 0, 0))

            # code for id printers
            cvzone.putTextRect(img, f"{int(id)} ", (max(0, x1), max(35, y1)), scale=1, thickness=0)
            # code for centers dot that moves along with the classes
            cx, cy = x1+w//2, y1+h//2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            # code for region where the object should be detected and count cars
            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-13 < cy < limitsUp[1]+13:
                if totalcountUp.count(id) == 0:
                    totalcountUp.append(id)
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)
            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-13 < cy < limitsDown[1]+13:
                if totalcountDown.count(id) == 0:
                    totalcountDown.append(id)
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)
        #cvzone.putTextRect(img, f" count: {len(totalcount)} ", (50, 50), scale=2, thickness=2)
        # printing of the car counters on the screen
        cvzone.putTextRect(img, str(len(totalcountUp)), (930, 330), 2, 3, (139, 195, 75), cv2.FONT_HERSHEY_PLAIN,)
        cvzone.putTextRect(img, str(len(totalcountDown)), (1191, 330 ), 2, 3, (50, 50, 230), cv2.FONT_HERSHEY_PLAIN, )
        cv2.imshow("Image", img)
        #cv2.imshow("imgRegion", imgRegion)
        cv2.waitKey(1)