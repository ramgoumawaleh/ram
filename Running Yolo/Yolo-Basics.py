from ultralytics import YOLO
import cv2

model = YOLO('../yolo-Weights/yolov8l.pt')
results = model("Images/image3.jpg",show=True)
cv2.waitKey(0)