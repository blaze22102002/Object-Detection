import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs\train\yolov8_custom6\weights\best.pt")  # Path to your saved model

# Open camera (0 for default camera, use 1, 2, etc. for other attached cameras)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Start capturing video and performing detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the frame
    results = model(frame,conf=0.7)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
