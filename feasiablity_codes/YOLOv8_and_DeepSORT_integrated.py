import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
from slowfast.utils import checkpointer
from slowfast.models import build_model
from slowfast.config.defaults import get_cfg
from PIL import Image

# Load YOLOv8 for object detection
model = YOLO("yolov8n.pt")  # Load YOLOv8

# Initialize DeepSORT for tracking
tracker = DeepSort()

# Initialize Action Recognition Model (e.g., SlowFast)
cfg = get_cfg()
cfg.merge_from_file("configs/Kinetics/SLOWFAST_8x8_R50.yaml")  # SlowFast config file
model_action = build_model(cfg)
checkpoint = checkpoint.Checkpoint(model_action)
checkpoint.load_checkpoint("models/slowfast_r50.pyth", None)  # Pre-trained SlowFast model
model_action.eval()

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

# FPS Calculation
prev_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame to RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect objects (people) using YOLOv8
    results = model(image)
    boxes = results.xywh[0][:, :4].cpu().numpy()  # Get bounding boxes
    confidences = results.xywh[0][:, 4].cpu().numpy()
    class_ids = results.xywh[0][:, 5].cpu().numpy()
    
    # Format detections for DeepSORT (x1, y1, x2, y2, confidence)
    detections = []
    for box, conf, cls in zip(boxes, confidences, class_ids):
        if cls == 0:  # Class ID 0 corresponds to 'person'
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf])

    # Update DeepSORT tracker with the new detections
    tracked_objects = tracker.update(np.array(detections))

    # Prepare the input for action recognition (SlowFast)
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
        tracked_person = frame[int(y1):int(y2), int(x1):int(x2)]

        # Preprocess the tracked person frame for action recognition
        tracked_person_tensor = transforms.ToTensor()(tracked_person).unsqueeze(0)
        tracked_person_tensor = tracked_person_tensor.cuda()  # Move to GPU

        # Get action prediction
        with torch.no_grad():
            actions = model_action(tracked_person_tensor)

        # Assume action is detected (use appropriate action index)
        action = "Walking"  # Just an example; use the predicted label here

        # Display tracked object ID and action
        cv2.putText(frame, f"ID: {int(track_id)} - Action: {action}", 
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw bounding box
    
    # FPS Calculation
    frame_count += 1
    curr_time = time.time()
    fps = frame_count / (curr_time - prev_time)

    # Display FPS on video feed
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show output frame
    cv2.imshow("Object Detection and Action Recognition", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
