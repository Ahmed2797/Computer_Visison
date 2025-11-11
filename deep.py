import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from final_.constant import *   
import json
import time

class VehicleCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(data_path)
        self.model = YOLO(model_path)

        self.mask = cv2.imread(mask_path)
        self.mask = cv2.resize(self.mask, (1080, 720))

        self.frame_count = 0
        self.skip = 1
        self.allowed_classes = allowed_classes
        self.line_y_red = line_y_red

        self.class_count = defaultdict(int)
        self.crossed_ids = set()
        
        # New features
        self.tracking_history = defaultdict(lambda: deque(maxlen=30))  # Track movement history
        self.count_log = []  # Log counts over time
        self.start_time = time.time()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)

    def process(self):
        while True:
            start_frame_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1080, 720))
            masked = cv2.bitwise_and(self.mask, frame)

            if self.frame_count % self.skip == 0:
                result = self.model.track(masked, persist=True, conf=0.3)  # Added confidence threshold

                if result[0].boxes is not None and result[0].boxes.id is not None:
                    self.draw_results(frame, result)
                    
            # Add FPS counter
            fps = self.calculate_fps(start_frame_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (50, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("YOLO Object Counter", frame)
            self.frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.save_results()
        self.cap.release()
        cv2.destroyAllWindows()

    def draw_results(self, frame, result):
        boxes = result[0].boxes.xyxy.cpu()
        confidence = result[0].boxes.conf.cpu()
        track_ids = result[0].boxes.id.int().cpu().tolist()
        classes = result[0].boxes.cls.int().cpu().tolist()

        # Draw counting line
        cv2.line(frame, (580, self.line_y_red), (990, self.line_y_red), (0, 0, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (600, self.line_y_red - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for box, conf, id, cls in zip(boxes, confidence, track_ids, classes):
            if cls not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box)
            label = self.model.names[cls]  

            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Update tracking history
            self.tracking_history[id].append((cx, cy))
            
            # Draw bounding box with class-specific colors
            color = self.get_class_color(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            cv2.putText(frame, f"ID:{id} {label} {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

            # Draw center point and tracking trail
            cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)
            self.draw_tracking_trail(frame, id)

            # Check line crossing with direction awareness
            if self.check_line_crossing(id, cy):
                self.crossed_ids.add(id)
                self.class_count[label] += 1
                self.log_count_event(label, id)

        # Draw statistics panel
        self.draw_statistics_panel(frame)

    def get_class_color(self, class_id):
        """Get consistent color for each class"""
        colors = {
            0: (255, 0, 0),    # car - blue
            1: (0, 255, 0),    # truck - green  
            2: (0, 0, 255),    # motorcycle - red
            3: (255, 255, 0),  # bus - cyan
        }
        return colors.get(class_id, (255, 255, 255))

    def draw_tracking_trail(self, frame, track_id):
        """Draw movement trail for tracked objects"""
        history = self.tracking_history[track_id]
        for i in range(1, len(history)):
            cv2.line(frame, history[i-1], history[i], (0, 255, 255), 2)

    def check_line_crossing(self, track_id, current_y):
        """Enhanced line crossing check with direction awareness"""
        if track_id in self.crossed_ids:
            return False
            
        history = self.tracking_history[track_id]
        if len(history) < 2:
            return False
            
        # Check if object crossed the line from above to below
        prev_y = history[-2][1] if len(history) >= 2 else current_y
        return prev_y <= self.line_y_red and current_y > self.line_y_red

    def draw_statistics_panel(self, frame):
        """Draw comprehensive statistics panel"""
        # Background for panel
        panel_height = 150
        cv2.rectangle(frame, (0, 0), (300, panel_height), (0, 0, 0), -1)
        
        y = 30
        total_count = sum(self.class_count.values())
        
        cv2.putText(frame, f"Total: {total_count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 25
        
        for name, count in self.class_count.items():
            cv2.putText(frame, f"{name}: {count}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.get_class_color_by_name(name), 2)
            y += 25
            
        # Add elapsed time
        elapsed_time = time.time() - self.start_time
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def get_class_color_by_name(self, class_name):
        """Get color by class name"""
        color_map = {
            "car": (255, 0, 0),
            "truck": (0, 255, 0),
            "motorcycle": (0, 0, 255),
            "bus": (255, 255, 0),
        }
        return color_map.get(class_name.lower(), (255, 255, 255))

    def calculate_fps(self, start_time):
        """Calculate and return current FPS"""
        frame_time = time.time() - start_time
        self.processing_times.append(frame_time)
        avg_time = np.mean(self.processing_times)
        return 1.0 / avg_time if avg_time > 0 else 0

    def log_count_event(self, class_name, track_id):
        """Log counting events for analysis"""
        event = {
            'timestamp': time.time() - self.start_time,
            'class': class_name,
            'track_id': track_id,
            'total_count': sum(self.class_count.values())
        }
        self.count_log.append(event)

    def save_results(self):
        """Save counting results to file"""
        results = {
            'total_count': sum(self.class_count.values()),
            'class_counts': dict(self.class_count),
            'count_log': self.count_log,
            'processing_time': time.time() - self.start_time
        }
        
        with open('counting_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to counting_results.json")

if __name__ == "__main__":
    counter = VehicleCounter()
    counter.process()