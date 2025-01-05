from ultralytics import YOLO
import cv2
import cvzone
import math

# Charger la vidéo
cap = cv2.VideoCapture("../Video/voiture.mp4")  # Remplacez par le chemin de votre vidéo

# Charger le modèle YOLO
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Définir la classe "car"
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "TVmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Indice pour la classe "car"
car_index = classNames.index("car")

# Variables pour compter les voitures
car_count = 0
line_position = 400  # Position de la ligne virtuelle (ajustez selon votre vidéo)
offset = 5  # Tolérance pour détecter les passages

# Liste pour suivre les objets déjà comptés
tracker = {}

while True:
    success, img = cap.read()
    if not success:
        break  # Sortir de la boucle si la vidéo est terminée

    # Redimensionner si nécessaire (facultatif)
    img = cv2.resize(img, (1280, 720))

    # Détection des objets
    results = model(img, stream=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extraire les coordonnées
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extraire la classe et la confiance
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Vérifier si l'objet est une voiture
            if cls == car_index and conf > 0.5:  # Seulement si la confiance est élevée
                # Dessiner la boîte englobante
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=5)

                # Dessiner le texte de la classe
                cvzone.putTextRect(img, f"Car {conf:.2f}", (x1, max(35, y1)), scale=0.7, thickness=1)

                # Calculer le centre de l'objet
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Dessiner le centre
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                # Vérifier si la voiture a traversé la ligne
                if line_position - offset < cy < line_position + offset:
                    if cx not in tracker:
                        car_count += 1
                        tracker[cx] = True  # Marquer l'objet comme compté

    # Dessiner la ligne virtuelle
    cv2.line(img, (0, line_position), (1280, line_position), (0, 255, 0), 3)

    # Afficher le nombre de voitures
    cv2.putText(img, f"Cars Counted: {car_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Car Counting", img)

    # Sortie avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
