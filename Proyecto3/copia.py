import os
import shutil

# Ruta de la imagen original
ruta_original = r"C:\Users\betor\Desktop\Shits\IA\Proyecto3\Murcielago\MurcielagoB.png"

# Carpeta donde se encuentra la imagen
carpeta_destino = os.path.dirname(ruta_original)

# Nombre base y extensión de la imagen
nombre_base, extension = os.path.splitext(os.path.basename(ruta_original))

# Crear 200 copias de la imagen
for i in range(1, 201):
    # Generar el nombre para cada copia
    nombre_copia = f"{nombre_base}_{i}{extension}"
    ruta_copia = os.path.join(carpeta_destino, nombre_copia)
    
    # Copiar la imagen
    shutil.copy(ruta_original, ruta_copia)

print("Se han creado 200 copias de la imagen en la misma carpeta.")
