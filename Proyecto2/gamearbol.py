import pygame
import random
import pandas as pd
import numpy as np
import graphviz
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os


#Pygame
pygame.init()
os.chdir(os.path.dirname(__file__))

w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
jugador = None
bala = None
fondo = None
nave = None
menu = None

salto = False
salto_altura = 15
gravedad = 1
en_suelo = True
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False
datos_modelo = []
model = None
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
menu_img = pygame.image.load('assets/game/menu.png')

fondo_img = pygame.transform.scale(fondo_img, (w, h))

jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

current_frame = 0
frame_speed = 10 
frame_count = 0


velocidad_bala = -10
bala_disparada = False
fondo_x1 = 0
fondo_x2 = w

def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-20, -10)
        bala_disparada = True


def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar bala
    bala_disparada = False

def reset_model():
    tf.keras.backend.clear_session()

def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura 
        salto_altura -= gravedad 

        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15   
            en_suelo = True

def manejar_autosalto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura 
        salto_altura -= gravedad  

        if jugador.y >= h - 100:
            jugador.y = h - 100  
            salto = False
            salto_altura = 15  
            en_suelo = True


def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    fondo_x1 -= 1
    fondo_x2 -= 1

    if fondo_x1 <= -w:
        fondo_x1 = w

    if fondo_x2 <= -w:
        fondo_x2 = w

    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    pantalla.blit(nave_img, (nave.x, nave.y))

    if bala_disparada:
        bala.x += velocidad_bala

    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()


# Guardar Datos/Graficar--------------------------------------------------------------------------------------------------------
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0 
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

def graficar_datos():

    x1 = [x for x, y, z in datos_modelo if z == 0]
    x2 = [y for x, y, z in datos_modelo if z == 0]
    target0 = [z for x, y, z in datos_modelo if z == 0]

    x3 = [x for x, y, z in datos_modelo if z == 1]
    x4 = [y for x, y, z in datos_modelo if z == 1]
    target1 = [z for x, y, z in datos_modelo if z == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x1, x2, target0, c='blue', marker='o', label='Target=0')
    ax.scatter(x3, x4, target1, c='red', marker='x', label='Target=1')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    ax.legend()

    plt.show()

def graficar_arbol():

    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    X = list(zip(x1, x2))
    y = target0 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Arbol----------------------------------------------------------------------------------------------------------------
    clf = DecisionTreeClassifier()

def train_tree():
    global model_tree
    print('entrenando con arbol')
    if len(datos_modelo) < 10:  #<------------- N Saltos para que jale
        print("No hay datos suficientes para entrenar el árbol de decisión.")
        return

    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    X = list(zip(x1, x2))
    y = target0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42) 
    model_tree = clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Precisión del Árbol de Decisión: {accuracy:.2f}")

# Entrenar----------------------------------------------------------------------------------------------------------------  
def train_model():
    global model

    if not datos_modelo:
        print("No hay datos suficientes para entrenar el modelo.")
        return

    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    X = np.array(list(zip(x1, x2)))
    y = np.array(target0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")
    
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

def mostrar_menu():
    global menu_activo, modo_auto, modo_auto_tree
    pantalla.fill(NEGRO)
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, 'G' para Graficar o 'Q' para Salir", False, BLANCO)
    pantalla.blit(texto, (10 , h // 2))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = True
                    menu_activo = False
                    if len(datos_modelo) < 10:
                        print("No hay suficientes datos para entrenar el árbol de decisión.")
                    else:
                        train_tree()  
                elif evento.key == pygame.K_m:
                    reset_model()
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_g:
                    modo_auto = False
                    menu_activo = False
                    graficar_arbol()
                    graficar_datos()
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True 
    jugador.x, jugador.y = 50, h - 100  
    bala.x = w - 50  
    nave.x, nave.y = w - 100, h - 100 
    bala_disparada = False
    salto = False
    en_suelo = True
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()  

def main():
    global salto, en_suelo, bala_disparada

    reloj = pygame.time.Clock()
    mostrar_menu()  
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  
                    pausa_juego()
                if evento.key == pygame.K_q:  
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            if modo_auto:
                distancia = abs(jugador.x - bala.x)
                velocidad = velocidad_bala

                entrada = np.array([[velocidad, distancia]])
                prediccion = model_tree.predict(entrada)[0] > 0.5

                if prediccion == 1 and en_suelo:
                    print("Salto solito")  
                    salto = True
                    en_suelo = False

            if salto:
                manejar_autosalto()

            if not modo_auto:
                if salto:
                    salto = True
                    en_suelo = False
                    manejar_salto()
                guardar_datos()

            if not bala_disparada:
                disparar_bala()
            update()

        pygame.display.flip()
        reloj.tick(30) 

    pygame.quit()

if __name__ == "__main__":
    main()