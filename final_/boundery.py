import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt') 
class_list = model.names
allowed_classes = [1, 2, 3, 5, 7, 9, 11]
class_name = {i: model.names[i] for i in allowed_classes}



cap = cv2.VideoCapture('4.mp4')
ret, frame = cap.read()
height, width = frame.shape[:2]

# Optional: reduce frame size
frame = cv2.resize(frame, (1080, 720))

# read_mask 
mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1080, 720))





skip = 6
frame_count = 0

pts = np.array([[456, 388],
                [760, 335],
                [456, 235],
                [620, 230]], np.int32)
pts = pts.reshape((-1, 1, 2))


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1080, 720))

    image_region = cv2.bitwise_and(mask,frame)
    if not ret:
        break

    if frame_count % skip == 0:
        result = model.track(image_region, persist=True)

        #cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        #cv2.rectangle(frame, (450,370), (740,245), (0, 200, 0), 2)
 

        if result[0].boxes.data is not None:
            boxes = result[0].boxes.xyxy.cpu()
            confidence = result[0].boxes.conf.cpu()
            track_id = result[0].boxes.id.int().cpu().tolist()
            classcs = result[0].boxes.cls.int().cpu().tolist()
            
       
            for box, conf, id, cls in zip(boxes, confidence, track_id, classcs):
                    x1, y1, x2, y2 = map(int, box)
                    cls = int(cls)
                    label = class_list[cls]  # class_list = model.names

                    #if label == 'car':  # detect only car
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(frame, (x1, y1, w, h), l=2)

                    cv2.putText(frame, f"ID:{id} {label}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 2)

       
        cv2.imshow("YOLO-Object-Detection", frame)
        #cv2.imshow('YOLO Object Detection',image_region)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
