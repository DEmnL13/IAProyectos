import os

def rename_files_in_folder(folder_path, base_name):
    try:
        # Verifica si la carpeta existe
        if not os.path.exists(folder_path):
            print(f"La carpeta '{folder_path}' no existe.")
            return

        # Lista todos los archivos en la carpeta
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Renombra cada archivo
        for index, file in enumerate(files, start=1):
            file_extension = os.path.splitext(file)[1]  # Obtiene la extensión del archivo
            new_name = f"{base_name}_{index}{file_extension}"
            old_file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(folder_path, new_name)

            os.rename(old_file_path, new_file_path)
            print(f"Renombrado: {file} -> {new_name}")

        print("Todos los archivos han sido renombrados correctamente.")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Especifica la carpeta y la palabra base para renombrar
folder_path = r"C:\Users\betor\Desktop\Shits\IA\Proyecto3\Quattro"
base_name = "Quattro" 

rename_files_in_folder(folder_path, base_name)