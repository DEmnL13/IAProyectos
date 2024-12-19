import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import SGD

# redimensionar imágenes
def redimensionar_imagenes(input_dir, output_dir, image_size=(32, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, filename)
                image = imread(filepath)
                resized_image = resize(image, image_size, anti_aliasing=True, preserve_range=True)
                output_path = os.path.join(output_dir, os.path.relpath(filepath, input_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                imsave(output_path, resized_image.astype("uint8"))
                print(f"Imagen redimensionada: {output_path}")

input_dir = "C:\\Users\\betor\\Desktop\\Shits\\IA\\Proyecto3\\Dataset"
output_dir = "C:\\Users\\betor\\Desktop\\Shits\\IA\\Proyecto3\\DatasetRe"
redimensionar_imagenes(input_dir, output_dir)

# Cargar el dataset
def cargar_dataset(dataset_dir):
    images = []
    labels = []
    class_names = []
    label = 0

    for class_dir in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            class_names.append(class_dir)
            for file_name in os.listdir(class_path):
                if file_name.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_path, file_name)
                    image = imread(image_path)
                    images.append(image)
                    labels.append(label)
            label += 1

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    return images, labels, class_names
#sacar imagenes
dataset_dir = "C:\\Users\\betor\\Desktop\\Shits\\IA\\Proyecto3\\DatasetRe"
X, y, class_names = cargar_dataset(dataset_dir)


train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

train_Y_one_hot = to_categorical(train_Y, num_classes=len(class_names))
valid_Y_one_hot = to_categorical(valid_Y, num_classes=len(class_names))
test_Y_one_hot = to_categorical(test_Y, num_classes=len(class_names))

#modelo
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(21, 28, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=1e-3, decay=1e-3 / 100),
              metrics=['accuracy'])

# entrenamiento
history = model.fit(train_X, train_Y_one_hot,
                    batch_size=64,
                    epochs=20,
                    verbose=1,
                    validation_data=(valid_X, valid_Y_one_hot))

# Guardar el modelo
model.save("modelo_autos.h5")

test_loss, test_accuracy = model.evaluate(test_X, test_Y_one_hot, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Graficar 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()