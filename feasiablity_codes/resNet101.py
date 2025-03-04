import cv2
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DETR model with ResNet-101 backbone and move to GPU
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm").to(device)

# OpenCV Video Capture (0 for webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process image and move tensors to GPU
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO format
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    # Draw bounding boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = map(int, box)
        class_name = model.config.id2label[label.item()]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {round(score.item(), 2)}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output frame
    cv2.imshow("DETR Object Detection (ResNet-101, GPU)", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

