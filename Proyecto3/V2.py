import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)
from tensorflow.keras.utils import to_categorical

data_dir = 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/Dataset'  
categories = ['Chevelle', 'Corolla', 'Corvette', 'Murcielago', 'Quattro'] 
img_size = 128 

def load_images(data_dir, categories, img_size):
    images, labels = [], []  # Inicia etiquetas------------------------------------------------------------------
    for idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"El directorio {path} no existe. Saltando...")
            continue
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = plt.imread(img_path)
                if img.shape[-1] == 3:  # Verificar si la imagen es RGB o no :c
                    img_resized = tf.image.resize(img, (img_size, img_size)).numpy()
                    images.append(img_resized)
                    labels.append(idx)
            except Exception as e:
                print(f"Error al procesar {img_name} en {path}: {e}")
    if not images or not labels:
        raise ValueError("No se pudieron cargar imágenes o etiquetas. Verifique las rutas y los datos.")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


X, y = load_images(data_dir, categories, img_size) #x son las cargadas, y son las etiquetas

X = X / 255.0 # aqui se normalizan

y_one_hot = to_categorical(y, num_classes=len(categories)) #aqui se transforman en one-hot, que es una binarización

# Dividir datos en entrenamiento y prueba a 80% y 20% respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),# Aplica un filtro convoluc. de 3x3 pixeles y 32 filtros y extrae caract.
    MaxPooling2D((2, 2)),#reduce tamaño
    BatchNormalization(),#normaliza

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Flatten(),#la transforma en un vector
    Dense(256, activation='relu'),#Clasifica
    Dropout(0.6),#Técnica para evitar el sobreajuste, desconectando aleatoriamente algunas neuronas.
    Dense(len(categories), activation='softmax') #Función de activación en la última capa que convierte las salidas en probabilidades.
])

# Compilar el modelo--------------------------------------------------------------------------------------------
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenar~
history = model.fit(X_train, y_train, epochs=13, validation_data=(X_test, y_test))

# Guardar~
model.save('car_model_cnn.h5')

# Evaluar~
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")


y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)  # Convertir etiquetas one-hot a etiquetas originales y asi
print(classification_report(y_test_labels, y_pred, target_names=categories))

# Graficar~
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.legend()

plt.show()


def predict_car_image(model_path, image_path):
    model = load_model(model_path)

    img = plt.imread(image_path)
    if img.shape[-1] == 3:  
        img_resized = tf.image.resize(img, (img_size, img_size)).numpy() / 255.0
        img_resized = np.expand_dims(img_resized, axis=0) 

        predictions = model.predict(img_resized)
        predicted_class = np.argmax(predictions)

        print(f"La imagen cargada se clasifica como: {categories[predicted_class]} (Confianza: {predictions[0][predicted_class] * 100:.2f}%)")
        return categories[predicted_class]
    else:
        print("La imagen debe ser RGB.")

predict_car_image('car_model_cnn.h5', 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/ChevelleTest.jpg')
predict_car_image('car_model_cnn.h5', 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/CorollaTest.jpg')
predict_car_image('car_model_cnn.h5', 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/CorvetteTest.jpg')
predict_car_image('car_model_cnn.h5', 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/MurcielagoTest.jpg')
predict_car_image('car_model_cnn.h5', 'C:/Users/betor/Desktop/Shits/IA/Proyecto3/QuattroTest.jpg')
