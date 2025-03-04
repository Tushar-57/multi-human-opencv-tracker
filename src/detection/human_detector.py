import cv2
import numpy as np

class HumanDetector:
    def __init__(self, model_path: str = "/Users/tusharsharma/Desktop/AILearningFolder/Other/Projects/main/proj_cv/multi-human-opencv-tracker/src/mobilenet_ssd.caffemodel", 
                config_path: str = "/Users/tusharsharma/Desktop/AILearningFolder/Other/Projects/main/proj_cv/multi-human-opencv-tracker/src/deploy.prototxt",
                confidence_threshold: float = 0.5):
        """Initialize the Human Detector with SSD MobileNet model.
        
        Args:
            model_path: Path to the Caffe model file
            config_path: Path to the model configuration file
            confidence_threshold: Minimum confidence for detection
        """
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect_humans(self, frame):
        """Detect humans in the input frame using SSD MobileNet.
        
        Args:
            frame: Input frame/image as numpy array
        
        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for each detected person
        """
        if frame is None or frame.size == 0:
            return []
        
        height, width = frame.shape[:2]
        # SSD MobileNet requires 300x300 RGB inputs
        # Scale factor = 0.007843 (1/127.5), size = 300x300, mean = 127.5
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        persons = []
        # SSD output format: [batch, class, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]
            
            # Class ID 15 represents person in SSD MobileNet
            if class_id == 15 and confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                # Ensure coordinates are within frame boundaries
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)
                persons.append((startX, startY, endX, endY, confidence))
                
        return persons

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Use webcam feed
    detector = HumanDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        persons = detector.detect_humans(frame)
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

