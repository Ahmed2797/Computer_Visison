import cv2
import numpy as np
from ultralytics import YOLO
from object_tracking_library.sort import Sort
import math

# -----------------------------------------------
# Load YOLOv8 pre-trained model
# -----------------------------------------------
# Initializes the YOLOv8 nano model ("yolov8n.pt"), which is lightweight and fast.
# This model will be used to detect vehicles in each video frame.
model = YOLO("yolov8n.pt")

# -----------------------------------------------
# Video capture setup
# -----------------------------------------------
# Loads the input video file for processing.
cap = cv2.VideoCapture("traffic.mp4")

# Retrieves video properties: width, height, and FPS (frames per second).
# These values are needed for scaling visuals and writing output video properly.
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------------------------
# Lane polygons
# -----------------------------------------------
# Defines two polygonal regions representing LEFT and RIGHT traffic lanes.
# Each polygon is defined by 4 points in the image.
# These regions help classify which lane a tracked vehicle belongs to.
left_pts  = np.array([[340,155], [15,600], [485,650], [480,160]])
right_pts = np.array([[540,195], [520,715], [768,710], [645,185]])

# -----------------------------------------------
# SORT tracker
# -----------------------------------------------
# Initializes the SORT tracking algorithm:
# - max_age: tracker keeps lost objects for up to 30 frames
# - min_hits: an object needs at least 3 detections to be confirmed
# - iou_threshold: bounding boxes need at least 0.3 IoU to be associated
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Frame counter for timing, animation, and speed estimation.
frame_count = 0

# Stores previous positions of tracked objects for speed calculation.
prev_positions = {}

# -----------------------------------------------
# Output video writer
# -----------------------------------------------
# Defines the output video file path and codec.
output_path = "traffic_analytics_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Creates a video writer to save processed frames with overlays and dashboards.
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------------------------
# Unique IDs and cumulative counts
# -----------------------------------------------
# A set that stores unique track IDs — prevents recounting the same vehicle.
total_car_ids = set()

# Stores cumulative (per-frame) counts for graph plotting.
cumulative_counts = {"car":[], "bus":[], "truck":[]}

# -----------------------------------------------
# Pulsing gradient box helper
# -----------------------------------------------
# This function draws an animated gradient bounding box around detected objects.
# Includes:
# - Pulsing effect using sine wave
# - Corner decorations
# - Optional track ID label
# - Expanding layered rectangles for styling
def draw_pulsing_gradient_box(img, top_left, bottom_right, frame_idx, track_id=None, 
                              color_base=(0,255,0), thickness=2, radius=5):

    # Create a copy to draw gradient layers on
    overlay = img.copy()

    # Extract coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Compute pulsing brightness using a sine-based oscillation
    pulse = 0.5 + 0.5 * math.sin(frame_idx / 5)

    # Number of expanding layers
    steps = 15

    # Create layered rectangles to form the gradient glow
    for i in range(steps):
        interp_color = (
            int(color_base[0]*(i/steps) + 50*(1-i/steps)),
            int(color_base[1]*(i/steps) + 50*(1-i/steps)),
            int(color_base[2]*(i/steps) + 50*(1-i/steps))
        )
        shift_x = int(i*radius/steps)
        shift_y = int(i*radius/steps)
        cv2.rectangle(overlay, (x1-shift_x, y1-shift_y), (x2+shift_x, y2+shift_y), interp_color, -1)

    # Blend overlay with original image to achieve glow transparency
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Offset for decorative corner shapes
    offset = radius // 2
    stroke_thickness = 2

    # Compute coordinates for the decorated corner points
    corners = {
        "top_left":    (x1 - offset + radius, y1 - offset + radius),
        "top_right":   (x2 + offset - radius, y1 - offset + radius),
        "bottom_left": (x1 - offset + radius, y2 + offset - radius),
        "bottom_right":(x2 - offset + radius, y2 + offset - radius)
    }

    # Draw corner line segments
    cv2.line(img, (corners["top_left"][0], y1), (corners["top_right"][0], y1), color_base, thickness)
    cv2.line(img, (corners["bottom_left"][0], y2), (corners["bottom_right"][0], y2), color_base, thickness)
    cv2.line(img, (x1, corners["top_left"][1]), (x1, corners["bottom_left"][1]), color_base, thickness)
    cv2.line(img, (x2, corners["top_right"][1]), (x2, corners["bottom_right"][1]), color_base, thickness)

    # Draw glowing corner circles
    for cx, cy in corners.values():
        cv2.circle(img, (cx, cy), radius, color_base, -1)
        cv2.circle(img, (cx, cy), radius, (255,255,255), stroke_thickness)

    # Optionally draw object ID label above the bounding box
    if track_id is not None:
        text_id = f"ID:{track_id}"
        (tw, th), _ = cv2.getTextSize(text_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1-30), (x1+tw+6, y1-10), (0,0,0), -1)
        cv2.putText(img, text_id, (x1+3, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_base, 2)

# -----------------------------------------------
# Main loop
# -----------------------------------------------
while True:
    # Read next frame from video
    ret, frame = cap.read()

    # Stop if no more frames
    if not ret:
        break

    # Increment frame counter
    frame_count += 1

    # --- Lane overlay visualization ---
    # Create a blank mask same size as frame
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Fill lane polygons with colors (visual guidance only)
    cv2.fillPoly(mask, [left_pts], (0,255,0))
    cv2.fillPoly(mask, [right_pts], (255,0,255))

    # Blend original frame with lane mask for semi-transparent effect
    overlayed = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

    # Draw small circles on lane polygon vertices for clarity
    for pt in left_pts:
        cv2.circle(overlayed, tuple(pt), 4, (255,255,255), 2)
        cv2.circle(overlayed, tuple(pt), 2, (0,255,0), -1)
    for pt in right_pts:
        cv2.circle(overlayed, tuple(pt), 4, (255,255,255), 2)
        cv2.circle(overlayed, tuple(pt), 2, (255,0,255), -1)

    # --- YOLO object detection ---
    # Run inference on the current frame (verbose disabled for speed)
    results = model(frame, verbose=False)

    # List to store detected bounding boxes in SORT format
    dets = []

    # Loop through YOLO detections
    for box in results[0].boxes:
        cls = int(box.cls[0])

        # Filter only vehicle classes (car, bus, truck, etc.)
        # YOLO class IDs used: 2, 5, 7, 3
        if cls in [2,5,7,3]:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            # Append coordinates and dummy confidence score (1.0 required by SORT)
            dets.append([x1,y1,x2,y2,1.0])

    # Convert detections to numpy array for SORT
    dets = np.array(dets) if len(dets) > 0 else np.empty((0,5))

    # Update tracker with current detections
    # Returns tracked objects with assigned unique IDs
    tracks = tracker.update(dets)

    # Per-frame lane-wise counters
    left_counts  = {"car":0,"bus":0,"truck":0}
    right_counts = {"car":0,"bus":0,"truck":0}

    # --- Track each vehicle ---
    for track in tracks:
        # Extract bounding box and track ID
        x1, y1, x2, y2, track_id = map(int, track)

        # Compute center point of tracked object
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        lane = None
        color=(0,255,255)

        # Determine which lane the object belongs to using point-to-polygon test
        if cv2.pointPolygonTest(left_pts,(cx,cy),False) >= 0:
            lane="left"; color=(0,255,0)
        elif cv2.pointPolygonTest(right_pts,(cx,cy),False) >= 0:
            lane="right"; color=(255,0,255)

        # Default class type is "car"
        cls_type = "car"

        # Match tracked object to detection class by checking overlap
        for box in results[0].boxes:
            x1b,y1b,x2b,y2b = map(int, box.xyxy[0])
            # If center is inside a YOLO box, assign class label
            if x1b <= cx <= x2b and y1b <= cy <= y2b:
                cls_map = {2:"car",5:"bus",7:"truck",3:"truck"}
                cls_type = cls_map.get(int(box.cls[0]), "car")
                break

        # Update lane counts
        if lane=="left":
            left_counts[cls_type] += 1
        elif lane=="right":
            right_counts[cls_type] += 1

        # Add unique track ID to the global set
        total_car_ids.add(track_id)

        # --- Speed estimation ---
        speed = 0
        if track_id in prev_positions:
            prev_cx, prev_cy = prev_positions[track_id]
            # Pixel distance moved since last frame
            dist = math.hypot(cx-prev_cx, cy-prev_cy)
            # Scaled value to approximate speed (not real km/h)
            speed = dist * 2

        # Store current position for next frame
        prev_positions[track_id] = (cx, cy)

        # Draw animated bounding box for the tracked vehicle
        draw_pulsing_gradient_box(overlayed, (x1,y1), (x2,y2), frame_count, track_id, color_base=color)

        # If movement detected, show speed label above box
        if speed > 0:
            speed_text = f"{int(speed)} km/h"
            (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(overlayed, (x1, y1-45), (x1+tw+6, y1-30), (0,0,0), -1)
            cv2.putText(overlayed, speed_text, (x1+3, y1-33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Update cumulative per-frame history for graph plotting ---
    cumulative_counts["car"].append(left_counts["car"]+right_counts["car"])
    cumulative_counts["bus"].append(left_counts["bus"]+right_counts["bus"])
    cumulative_counts["truck"].append(left_counts["truck"]+right_counts["truck"])

    # Reset history if it grows too long (prevents memory buildup)
    if frame_count > 2000:
        cumulative_counts["car"].clear()
        cumulative_counts["bus"].clear()
        cumulative_counts["truck"].clear()
        pass

    # --- Dashboard Layout Calculations ---
    dash_w, dash_h = 300, 650
    dash_x1 = width - dash_w - 20
    dash_y1 = 20
    dash_x2 = width - 20
    dash_y2 = dash_y1 + dash_h

    # Draw shadow for 3D effect
    shadow = overlayed.copy()
    cv2.rectangle(shadow, (dash_x1+4,dash_y1+4),(dash_x2+4,dash_y2+4),(0,0,0),-1)
    overlayed = cv2.addWeighted(shadow, 0.25, overlayed, 0.75, 0)

    # Draw semi-transparent dashboard background
    dashboard = overlayed.copy()
    cv2.rectangle(dashboard,(dash_x1,dash_y1),(dash_x2,dash_y2),(0,0,0),-1)
    overlayed = cv2.addWeighted(dashboard,0.45,overlayed,0.55,0)
    font=cv2.FONT_HERSHEY_SIMPLEX

    # Blinking "Live Lane Analytics" title every 20 frames
    if frame_count % 20 < 10:
        cv2.putText(overlayed,"Live Lane Analytics",(dash_x1+20,dash_y1+30),font,0.7,(255,255,255),2)
    cv2.line(overlayed,(dash_x1+10,dash_y1+40),(dash_x2-10,dash_y1+40),(255,255,255),1)

    # --- Lane Legends / Per-Lane Stats ---
    box_w, box_h = 22, 22
    left_y = dash_y1+50
    right_y = left_y+100

    # Left lane indicator box + stats
    cv2.rectangle(overlayed,(dash_x1+15,left_y),(dash_x1+15+box_w,left_y+box_h),(0,255,0),-1)
    cv2.rectangle(overlayed,(dash_x1+15,left_y),(dash_x1+15+box_w,left_y+box_h),(255,255,255),2)
    cv2.putText(overlayed,"Left Lane",(dash_x1+50,left_y+18),font,0.6,(255,255,255),2)
    cv2.putText(overlayed,f"Cars: {left_counts['car']}",(dash_x1+50,left_y+45),font,0.5,(255,255,255),2)
    cv2.putText(overlayed,f"Buses: {left_counts['bus']}",(dash_x1+50,left_y+65),font,0.5,(255,255,255),2)
    cv2.putText(overlayed,f"Trucks: {left_counts['truck']}",(dash_x1+50,left_y+85),font,0.5,(255,255,255),2)

    # Right lane indicator box + stats
    cv2.rectangle(overlayed,(dash_x1+15,right_y),(dash_x1+15+box_w,right_y+box_h),(255,0,255),-1)
    cv2.rectangle(overlayed,(dash_x1+15,right_y),(dash_x1+15+box_w,right_y+box_h),(255,255,255),2)
    cv2.putText(overlayed,"Right Lane",(dash_x1+50,right_y+20),font,0.6,(255,255,255),2)
    cv2.putText(overlayed,f"Cars: {right_counts['car']}",(dash_x1+50,right_y+47),font,0.5,(255,255,255),2)
    cv2.putText(overlayed,f"Buses: {right_counts['bus']}",(dash_x1+50,right_y+65),font,0.5,(255,255,255),2)
    cv2.putText(overlayed,f"Trucks: {right_counts['truck']}",(dash_x1+50,right_y+85),font,0.5,(255,255,255),2)

    # --- Progress bars representing lane distribution ---
    bar_x = dash_x1 + 20
    bar_w_max = dash_w - 40
    bar_h = 20

    # Used for scaling bar lengths
    max_total = max(sum(left_counts.values()), sum(right_counts.values()), 1)

    y_start = dash_y1 + 275

    # Left lane progress bars
    cv2.putText(overlayed,"Left Lane",(bar_x, y_start - 15),font,0.5,(255,255,255),2)
    cv2.rectangle(overlayed, (bar_x, y_start), (bar_x + int(left_counts["car"]/max_total*bar_w_max), y_start + bar_h), (0,255,0), -1)
    cv2.rectangle(overlayed, (bar_x, y_start + bar_h + 5), (bar_x + int(left_counts["bus"]/max_total*bar_w_max), y_start + 2*bar_h + 5), (0,128,255), -1)
    cv2.rectangle(overlayed, (bar_x, y_start + 2*(bar_h+5)), (bar_x + int(left_counts["truck"]/max_total*bar_w_max), y_start + 3*bar_h + 10), (0,0,255), -1)
    cv2.rectangle(overlayed, (bar_x, y_start), (bar_x + bar_w_max, y_start + 3*bar_h + 10), (255,255,255), 1)

    # Right lane progress bars
    y_start_right = dash_y1 + 385
    cv2.putText(overlayed,"Right Lane",(bar_x, y_start_right - 15),font,0.5,(255,255,255),2)
    cv2.rectangle(overlayed, (bar_x, y_start_right), (bar_x + int(right_counts["car"]/max_total*bar_w_max), y_start_right + bar_h), (0,255,0), -1)
    cv2.rectangle(overlayed, (bar_x, y_start_right + bar_h + 5), (bar_x + int(right_counts["bus"]/max_total*bar_w_max), y_start_right + 2*bar_h + 5), (0,128,255), -1)
    cv2.rectangle(overlayed, (bar_x, y_start_right + 2*(bar_h+5)), (bar_x + int(right_counts["truck"]/max_total*bar_w_max), y_start_right + 3*bar_h + 10), (0,0,255), -1)
    cv2.rectangle(overlayed, (bar_x, y_start_right), (bar_x + bar_w_max, y_start_right + 3*bar_h + 10), (255,255,255), 1)

    # -----------------------------------------------
    # Live cumulative count graph
    # -----------------------------------------------
    # Defines graph area in dashboard
    graph_h = 60
    graph_w = dash_w - 40
    graph_y = dash_y2 - graph_h - 120
    graph_x = dash_x1 + 20

    # Background for graph
    cv2.rectangle(overlayed, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h), (30,30,30), -1)

    # Highest cumulative count used to scale graph height
    max_count = max(max(cumulative_counts["car"]), max(cumulative_counts["bus"]), max(cumulative_counts["truck"]),1)

    # Plot line segments for each vehicle type
    for i in range(1, len(cumulative_counts["car"])):
        x1 = graph_x + int((i-1)/len(cumulative_counts["car"])*graph_w)
        x2 = graph_x + int(i/len(cumulative_counts["car"])*graph_w)

        # Car cumulative trend
        cv2.line(overlayed, (x1, graph_y+graph_h-int(cumulative_counts["car"][i-1]/max_count*graph_h)),
                 (x2, graph_y+graph_h-int(cumulative_counts["car"][i]/max_count*graph_h)), (0,255,0), 2)

        # Bus cumulative trend
        cv2.line(overlayed, (x1, graph_y+graph_h-int(cumulative_counts["bus"][i-1]/max_count*graph_h)),
                 (x2, graph_y+graph_h-int(cumulative_counts["bus"][i]/max_count*graph_h)), (0,128,255), 2)

        # Truck cumulative trend
        cv2.line(overlayed, (x1, graph_y+graph_h-int(cumulative_counts["truck"][i-1]/max_count*graph_h)),
                 (x2, graph_y+graph_h-int(cumulative_counts["truck"][i]/max_count*graph_h)), (0,0,255), 2)

    # -----------------------------------------------
    # Total stats
    # -----------------------------------------------
    # Per-frame total from both lanes
    total_vehicles = sum(left_counts.values()) + sum(right_counts.values())

    # Number of unique tracked vehicles
    total_unique_cars = len(total_car_ids)

    # Synthetic average speed (visual dummy metric)
    avg_speed = round(30 + math.sin(frame_count/5)*10)

    # Display summary stats at bottom of dashboard
    cv2.putText(overlayed,f"Total Cars Counted: {total_unique_cars}",(dash_x1+20,dash_y2-80),font,0.6,(255,255,255),2)
    cv2.putText(overlayed,f"Total Lane Buses: {left_counts['bus']+right_counts['bus']}",(dash_x1+20,dash_y2-50),font,0.6,(255,255,255),2)
    cv2.putText(overlayed,f"Total Lane Trucks: {left_counts['truck']+right_counts['truck']}",(dash_x1+20,dash_y2-20),font,0.6,(255,255,255),2)

    # Write processed frame to output video file
    out.write(overlayed)

    # Display speed control & quit on 'q'
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

# Cleanup: Release video resources and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Notify user where output is saved
print(f"Video saved to {output_path}")
