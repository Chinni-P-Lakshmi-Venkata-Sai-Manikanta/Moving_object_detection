#!/usr/bin/env python3
"""
Working Object Detection System
Real-time detection with camera input
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

class WorkingDetector:
    """
    Working object detection system with real-time feedback
    """
    
    def __init__(self):
        print("🔧 Initializing Working Detector...")
        
        # Initialize YOLO model
        try:
            self.model = YOLO('yolov8s.pt')
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            self.model = None
        
        # Detection parameters
        self.confidence_threshold = 0.25  # Lower threshold for better detection
        self.nms_threshold = 0.4
        
        # Performance tracking
        self.fps_counter = []
        self.detection_count = 0
        self.total_detections = 0
        
    def detect_objects(self, frame):
        """Detect objects in frame"""
        if self.model is None:
            return []
        
        try:
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[class_id]
                            
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detection results on frame"""
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Choose color based on class
            if 'person' in class_name.lower():
                color = (0, 255, 0)  # Green for person
            elif 'car' in class_name.lower() or 'truck' in class_name.lower():
                color = (255, 0, 0)  # Red for vehicles
            elif 'bicycle' in class_name.lower() or 'motorcycle' in class_name.lower():
                color = (0, 0, 255)  # Blue for bikes
            else:
                color = (255, 255, 0)  # Yellow for others
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw detection number
            cv2.putText(frame, f"#{i+1}", (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_info_panel(self, frame, detections, fps=0):
        """Draw comprehensive info panel"""
        # Create info panel
        panel_width = 350
        panel_height = 120
        panel_x = 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), (255, 255, 255), 2)
        
        # Draw info text
        y_offset = panel_y + 25
        line_height = 20
        
        # Objects detected
        cv2.putText(frame, f"Objects Detected: {len(detections)}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Total detections
        cv2.putText(frame, f"Total Detections: {self.total_detections}", 
                   (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Detection list
        if detections:
            detection_text = ", ".join([det['class_name'] for det in detections[:3]])
            if len(detections) > 3:
                detection_text += "..."
            cv2.putText(frame, f"Classes: {detection_text}", 
                       (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def draw_instructions(self, frame):
        """Draw control instructions"""
        instructions = [
            "Q - Quit",
            "S - Save frame", 
            "R - Reset counter",
            "C - Change confidence"
        ]
        
        x = frame.shape[1] - 150
        y = 30
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (x, y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

def run_detection():
    """Run the detection system"""
    print("🚀 Working Object Detection System")
    print("=" * 50)
    
    detector = WorkingDetector()
    
    if detector.model is None:
        print("❌ Cannot start detection - YOLO model not loaded")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    # Set camera properties for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
    
    print("✅ Camera initialized")
    print("🎯 Detection started - Look for objects!")
    print("Controls: Q=Quit, S=Save, R=Reset, C=Confidence")
    
    fps_counter = []
    frame_count = 0
    last_detection_time = 0
    
    while True:
        start_time = time.time()
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Update counters
        if detections:
            detector.total_detections += len(detections)
            last_detection_time = time.time()
        
        # Draw results
        result_frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        fps_counter.append(processing_time)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        
        fps = 1.0 / np.mean(fps_counter) if fps_counter else 0
        
        # Draw info panel
        result_frame = detector.draw_info_panel(result_frame, detections, fps)
        result_frame = detector.draw_instructions(result_frame)
        
        # Add detection status
        if detections:
            cv2.putText(result_frame, "DETECTION ACTIVE!", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(result_frame, "No objects detected", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Working Object Detection", result_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_frame_{frame_count}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"✅ Frame saved as {filename}")
        elif key == ord('r'):
            detector.total_detections = 0
            print("🔄 Counter reset")
        elif key == ord('c'):
            # Toggle confidence threshold
            if detector.confidence_threshold == 0.25:
                detector.confidence_threshold = 0.5
                print("🔧 Confidence threshold: 0.5")
            elif detector.confidence_threshold == 0.5:
                detector.confidence_threshold = 0.7
                print("🔧 Confidence threshold: 0.7")
            else:
                detector.confidence_threshold = 0.25
                print("🔧 Confidence threshold: 0.25")
        
        frame_count += 1
        
        # Print detection info every 60 frames
        if frame_count % 60 == 0:
            print(f"Frame {frame_count}: {len(detections)} objects, FPS: {fps:.1f}, Total: {detector.total_detections}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total detections: {detector.total_detections}")
    print(f"  Average FPS: {fps:.1f}")
    print("✅ Detection completed")

if __name__ == "__main__":
    run_detection()
