# pip install ultralytics supervision supervision[tracking]



## Image Detection
import cv2
from ultralytics import YOLO
import supervision as sv

# 1. Load YOLO model
model = YOLO("yolov8n.pt")   # ছোট, ফাস্ট মডেল

# 2. Read image
image = cv2.imread("car.jpg")

# 3. Run YOLO prediction
results = model(image)[0]

# 4. Convert YOLO → Supervision detections
detections = sv.Detections.from_ultralytics(results)
detections = sv.Detections(
    xyxy=results[0].boxes.xyxy.cpu().numpy(),
    confidence=results[0].boxes.conf.cpu().numpy(),
    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
)


# 5. Create annotator
box_annotator = sv.BoxAnnotator()

# 6. Draw bounding boxes
annotated = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# 7. Show output
cv2.imshow("YOLO Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()


## video detection
import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")

# input video
video_path = "input.mp4"
cap = cv2.VideoCapture(video_path)

box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model(frame)[0]

    # Convert to supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # Annotate
    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    cv2.imshow("Video Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()



## object detection
import cv2
from ultralytics import YOLO
import supervision as sv
from supervision.tracker.byte_tracker import ByteTrack

# Load YOLO
model = YOLO("yolov8n.pt")

# Video
cap = cv2.VideoCapture("input.mp4")

# Tracker
tracker = ByteTrack()

# Annotator
box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO prediction
    results = model(frame)[0]

    # supervision format
    detections = sv.Detections.from_ultralytics(results)

    # run tracking
    tracked = tracker.update(detections)

    # Draw boxes
    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=tracked
    )

    cv2.imshow("Object Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


