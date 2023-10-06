import cv2
import os
from PIL import Image
import numpy as np

# Fonction pour charger les visages enregistrés
def charger_visages_enregistres(dossier="photos"):
    visages_enregistres = {}
    for fichier in os.listdir(dossier):
        if fichier.endswith(".jpg"):
            prenom = os.path.splitext(fichier)[0]
            chemin_image = os.path.join(dossier, fichier)
            image = Image.open(chemin_image).convert("RGB")
            visages_enregistres[prenom] = np.array(image)
    return visages_enregistres

# Fonction pour comparer le visage détecté avec les visages enregistrés
def comparer_visages(visage_detecte, visages_enregistres):
    similarite_max = 0.0
    personne_reconnue = None

    for prenom, visage_enregistre in visages_enregistres.items():
        # Convertir en niveaux de gris pour la comparaison
        visage_detecte_gris = cv2.cvtColor(visage_detecte, cv2.COLOR_BGR2GRAY)
        visage_enregistre_gris = cv2.cvtColor(visage_enregistre, cv2.COLOR_BGR2GRAY)

        # Redimensionner le visage enregistré pour correspondre à la taille du visage détecté
        visage_enregistre_gris = cv2.resize(visage_enregistre_gris, (visage_detecte_gris.shape[1], visage_detecte_gris.shape[0]))

        # Calculer la similarité entre les deux visages
        difference = cv2.absdiff(visage_detecte_gris, visage_enregistre_gris)
        similarite = np.mean(difference)

        # Mettre à jour la personne reconnue si la similarité est élevée
        if similarite < 50 and (similarite_max == 0.0 or similarite < similarite_max):
            similarite_max = similarite
            personne_reconnue = prenom

    return personne_reconnue, similarite_max

# Initialiser le détecteur de visage
detecteur_visage = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Charger les visages enregistrés
visages_enregistres = charger_visages_enregistres()

# Initialiser la capture vidéo
capture = cv2.VideoCapture(0)

while True:
    # Lire un cadre de la vidéo
    ret, frame = capture.read()

    # Convertir l'image en niveaux de gris
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    visages = detecteur_visage.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

    # Afficher les visages reconnus en temps réel
    for (x, y, w, h) in visages:
        # Récupérer le visage à partir du cadre
        visage_detecte = frame[y:y+h, x:x+w]

        # Comparer le visage détecté avec les visages enregistrés
        personne_reconnue, _ = comparer_visages(visage_detecte, visages_enregistres)

        # Afficher un carré autour du visage détecté
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Afficher le nom de la personne reconnue
        if personne_reconnue:
            cv2.putText(frame, personne_reconnue, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afficher la vidéo en temps réel
    cv2.imshow('Video', frame)

    # Quitter la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et détruire toutes les fenêtres OpenCV
capture.release()
cv2.destroyAllWindows()
