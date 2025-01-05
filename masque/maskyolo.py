from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)  # pour webcam
cap.set(3, 1280)
cap.set(4, 780)

# cap = cv2.VideoCapture("../Video/ttttt.mp4")  # pour vidéo

model = YOLO("best.pt")

classNames = ['with_mask', 'without_mask', 'mask_weared_incorrect']

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bouding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # class names
            cls = int(box.cls[0])

            # Set the color based on the class
            if classNames[cls] == 'without_mask':
                color = (0, 0, 255)  # Red for 'without_mask'
            elif classNames[cls] == 'mask_weared_incorrect':
                color = (255, 0, 255)  # Violet for 'mask_weared_incorrect'
            else:
                color = (0, 255, 0)  # Green for 'with_mask'

            # Display the text with the appropriate color
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1,
                               colorR=color)

    # Show the image and check if 'q' is pressed
    cv2.imshow("Image", img)

    # Wait for a key press and quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
