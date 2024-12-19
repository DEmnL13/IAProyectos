import os
from PIL import Image

# Ruta del directorio que contiene las imágenes
directorio = r"c:\Users\betor\Desktop\Shits\IA\Proyecto3"

# Dimensiones deseadas para las imágenes
ancho, alto = 64, 32

# Recorrer todos los archivos en el directorio
for archivo in os.listdir(directorio):
    ruta_archivo = os.path.join(directorio, archivo)
    
    # Verificar si el archivo es una imagen (por extensión)
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # Abrir la imagen
            with Image.open(ruta_archivo) as img:
                # Redimensionar la imagen
                img_redimensionada = img.resize((ancho, alto))
                
                # Guardar la imagen redimensionada en el mismo lugar (sobrescribe el original)
                img_redimensionada.save(ruta_archivo)
                print(f"Redimensionada: {archivo}")
        except Exception as e:
            print(f"No se pudo procesar {archivo}: {e}")

print("Todas las imágenes en el directorio han sido redimensionadas a 64x32.")
