import cv2
import time
from ultralytics import YOLO
import torch

# Check if GPU (MPS or CUDA) is available and set device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model (which is more efficient for real-time use)
model = YOLO("yolov8n.pt")  # Small YOLOv8 (you can try larger versions like yolov8s.pt or yolov8m.pt)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# FPS Calculation
prev_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image to smaller resolution to speed up processing
    frame_resized = cv2.resize(frame, (640, 480))  # You can try different sizes

    # Run inference
    results = model(frame_resized)

    # Extract bounding boxes and labels
    boxes = results[0].boxes.xywh  # [x, y, width, height]
    confidences = results[0].boxes.conf
    labels = results[0].boxes.cls

    # Draw bounding boxes
    for i, box in enumerate(boxes):
        if confidences[i] > 0.4:  # Filter low-confidence detections
            x1, y1, w, h = map(int, box)
            x2, y2 = x1 + w, y1 + h
            class_name = model.names[labels[i].item()]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidences[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS Calculation
    frame_count += 1
    curr_time = time.time()
    fps = frame_count / (curr_time - prev_time)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show output frame
    cv2.imshow("YOLOv8 Object Detection (FPS)", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
