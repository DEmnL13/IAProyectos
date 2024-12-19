import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model


def predecir_imagenes(rutas_imagenes, modelo, class_names, image_size=(21, 28)):
    test_images = []
    
    for image_path in rutas_imagenes:
        image = imread(image_path)
        image_resized = resize(image, image_size, anti_aliasing=True, preserve_range=True)
        test_images.append(image_resized)

    test_images = np.array(test_images, dtype=np.float32) / 255.0  # iguialar imágenes

    predictions = modelo.predict(test_images)

    # resultados
    for i, pred in enumerate(predictions):
        print(f"Imagen: {rutas_imagenes[i]} - Predicción: {class_names[np.argmax(pred)]}")


modelo_cargado = load_model("modelo_autos.h5")

#lista
class_names = ["Chevelle",  "Corolla", "Corvette","Murcielago","Quattro"]

rutas_nuevas_imagenes = [
    "C:\\Users\\betor\\Desktop\\Shits\\IA\\Proyecto3\\QuattroTest.jpg",
    "C:\\Users\\betor\\Desktop\\Shits\\IA\\Proyecto3\\CorvetteTest.jpg."
    
]  


predecir_imagenes(rutas_nuevas_imagenes, modelo_cargado, class_names)