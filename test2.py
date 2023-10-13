import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

def detect_face(image_path):
    # Charger l'image
    img = cv2.imread(image_path)
    # Convertir l'image en RGB (car OpenCV charge les images en BGR par défaut)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Utiliser DeepFace pour la détection faciale
    faces = DeepFace.detectFace(img_rgb, detector_backend='opencv')
    
    # Afficher l'image avec les visages détectés
    plt.imshow(cv2.cvtColor(faces[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    photos_dir = "photos"  # Dossier contenant les images
    image_files = os.listdir(photos_dir)
    
    # Parcourir toutes les images dans le dossier
    for image_file in image_files:
        image_path = os.path.join(photos_dir, image_file)
        detect_face(image_path)
