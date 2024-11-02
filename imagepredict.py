import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO


model = YOLO("runs/train/yolov8_custom6/weights/best.pt")  


image_path = "3-Button-1200-DPI-USB-Wired-Mouse-Silent-Optical-Gaming-Mouses-For-PC-Laptop-Wholesale-20A_jpg.rf.31dd7b88ef575dd74a2f84b22b8ce0a9.jpg" 
image_pil = Image.open(image_path)
image_pil = ImageOps.exif_transpose(image_pil) 
image = np.array(image_pil)  


max_width, max_height = 640, 640 
image = cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)


results = model(image,conf=0.45)  


annotated_image = results[0].plot()


cv2.imshow("YOLOv8 Object Detection", annotated_image)


cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite("annotated_image.jpg", annotated_image)
