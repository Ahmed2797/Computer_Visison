import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11n.pt')   # Make sure your model path is correct
class_list = model.names

# Open video
cap = cv2.VideoCapture('live_PPE_detection.mp4')
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit()

# Original frame size
height, width = frame.shape[:2]

# Resize for faster processing
new_width, new_height = 840, 540
frame = cv2.resize(frame, (new_width, new_height))

# Video writer
# out = cv2.VideoWriter("fast_video.mp4",
#                       cv2.VideoWriter_fourcc(*'mp4v'),
#                       60,
#                       (new_width, new_height))

# Skip frames for speed
skip = 5
frame_count = 0

# Original polygon points
pts = np.array([[460, 420],
                [880, 390],
                [1185, 365],
                [1490, 600]], np.int32)

# Scale polygon points according to resized frame
scale_x = new_width / width
scale_y = new_height / height
pts_scaled = np.array([[int(x*scale_x), int(y*scale_y)] for x, y in pts.reshape(-1, 2)], np.int32).reshape((-1,1,2))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % skip == 0:
        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Run YOLO tracking
        results = model.track(frame, persist=True)

        
        # Draw YOLO boxes
        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, conf, tid, cls in zip(boxes, confidences, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                class_name = class_list[cls]
                cv2.circle(frame,(cx,cy),4,(20,0,250),2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"ID:{tid} {class_name} {conf:.2f}", 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        # Write frame to video
        # out.write(frame)

        # Show frame (local only; in Colab use cv2_imshow)
        cv2.imshow("YOLO Object Detection", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
