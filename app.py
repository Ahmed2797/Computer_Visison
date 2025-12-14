import cv2
import numpy as np
from ultralytics import YOLO 
from collections import defaultdict

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

class_count = defaultdict(int)
crossed_ids = set()
line_y_red = 460

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1080, 720))

    direction = np.empty((0,5))

    image_region = cv2.bitwise_and(mask,frame)
    if not ret:
        break

    if frame_count % skip == 0:
        result = model.track(image_region, persist=True)

        if result[0].boxes.data is not None:
            boxes = result[0].boxes.xyxy.cpu()
            confidence = result[0].boxes.conf.cpu()
            track_id = result[0].boxes.id.int().cpu().tolist()
            classcs = result[0].boxes.cls.int().cpu().tolist()

            cv2.line(frame,(580,line_y_red),(990,line_y_red),(0,0,200),2)
            
       
            for box, conf, id, cls in zip(boxes, confidence, track_id, classcs):
                x1, y1, x2, y2 = map(int, box)
                cls = int(cls)
                label = class_list[cls]  # class_list = model.names

                #if label == 'car':  # detect only car
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                cv2.circle(frame,(cx,cy),2,(20,0,250),-1)

                cv2.putText(frame, f"ID: {id}  {label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 200), 2)
                
                if cy > line_y_red and id not in crossed_ids:
                    crossed_ids.add(id)
                    class_count[label] += 1
                
            y_upset = 30
            for class_name,count in class_count.items():
                cv2.putText(frame, f"{class_name} {count}", (50,y_upset),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 200), 2)
                y_upset += 30



        cv2.imshow("YOLO-Object-Detection", frame)
        #cv2.imshow('YOLO Object Detection',image_region)

    frame_count += 2
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
