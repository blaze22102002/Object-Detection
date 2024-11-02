import cv2
from ultralytics import YOLO

model = YOLO(r"runs\train\yolov8_custom6\weights\best.pt")  


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break


    results = model(frame,conf=0.7)


    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
