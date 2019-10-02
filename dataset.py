import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = "TrashBot-Dataset"  # Cartella Dataset
CATEGORIES = ["bottigliette", "cartacce", "lattine"]  # Tutte le categorie dei dati
IMG_SIZE = 100    # Ridimensionamento foto

training_data = []   # Array principale dove ci sono tutti i dati


# Funzione per riempire l'array
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Percorso per le foto varie
        class_num = CATEGORIES.index(category)  # Stabiliamo a quanto corrisponde bottigliette, cartacce e lattine
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)   # Creiamo l'array con tutte le foto
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))                 # Le riduciamo di qualità
                training_data.append([new_array, class_num])   # Le aggiungiamo all'array
            except Exception as e:
                pass


create_training_data()

X = []
Y = []

# Dividiamo training_data in due array
for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y)

