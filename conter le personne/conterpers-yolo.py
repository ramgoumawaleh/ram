import cv2
import math
from ultralytics import YOLO

# Charger la vidéo
cap = cv2.VideoCapture("../Video/personne2.mp4")  # Remplacez par le chemin de votre vidéo

# Charger le modèle YOLO
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Définir les lignes pour les ascenseurs (basées sur la résolution 1920x1080)
line_up = [(950, 400), (1050, 400)]  # Ligne supérieure (ascenseur vers le haut)
line_down = [(950, 700), (1050, 700)]  # Ligne inférieure (ascenseur vers le bas)

# Variables pour le comptage
up_count = 0
down_count = 0

# Fonction pour vérifier si un point croise une ligne
def is_crossing_line(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Calculer la position relative du point par rapport à la ligne
    cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return cross

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1920, 1080))

    # Détection avec YOLO
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Extraire les coordonnées du bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Calculer le centre de la box

            # Vérifier si c'est une personne
            class_id = int(box.cls[0])
            if class_id == 0:  # Classe "person" dans COCO
                # Dessiner le bounding box et le centre
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Vérifier les traversées des lignes
                if is_crossing_line((cx, cy), line_up[0], line_up[1]) < 0:
                    up_count += 1
                elif is_crossing_line((cx, cy), line_down[0], line_down[1]) > 0:
                    down_count += 1

    # Dessiner les lignes
    cv2.line(frame, line_up[0], line_up[1], (0, 255, 0), 3)   # Ligne verte (montée)
    cv2.line(frame, line_down[0], line_down[1], (0, 0, 255), 3)  # Ligne rouge (descente)

    # Afficher les compteurs
    cv2.putText(frame, f'Up Count: {up_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Down Count: {down_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher la vidéo
    cv2.imshow("People Counter", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
