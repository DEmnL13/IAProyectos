## Proyecto Final
+ Proyecto1:
Este código implementa un simulador visual del algoritmo A* con soporte para rutas diagonales. A grandes rasgos, realiza lo siguiente:
- Crea una cuadrícula (grid):
Cada celda es un nodo representado por la clase Node. Los nodos tienen propiedades como si son transitables (walkable), su posición, y conexiones con nodos vecinos.
- Interactividad con el usuario:
Dibujar paredes (nodos no transitables) con clic izquierdo.
Seleccionar el nodo inicial con clic derecho.
Seleccionar el nodo final con clic central.
Presionar Espacio para ejecutar el algoritmo A* desde el nodo inicial al nodo final.
- Ejecución del algoritmo A*:
Calcula el camino más corto desde un nodo inicial hasta uno final usando una combinación de una cola de prioridad (heapq) y una heurística (distancia de Manhattan/diagonal).
Marca nodos explorados y visualiza el proceso en tiempo real, coloreando:
Nodos explorados en amarillo.
El nodo actual en verde.
El nodo inicial y final en rojo.
La ruta más corta en azul.

+ Proyecto2:
Este código es un juego en 2D desarrollado con Pygame, que incluye características de inteligencia artificial y aprendizaje automático.
El juego consiste en un personaje que puede saltar para evitar balas disparadas automáticamente por una nave. 
Los elementos clave son: la mecánica del salto controlado por gravedad y altura, el movimiento de fondo para simular desplazamiento, y la 
detección de colisiones entre el personaje y las balas. Además, el juego registra datos relacionados con la velocidad de las balas, 
la distancia al jugador y si este salta, para entrenar modelos de machine learning (árboles de decisión y redes neuronales) que pueden predecir 
cuándo el personaje debería saltar automáticamente. El sistema permite alternar entre modos manual y automático, pausar el juego, y visualizar datos mediante gráficos. 
También se incluye un menú inicial con opciones para configurar el modo de juego.

+ Proyecto3:
Este código implementa un sistema de clasificación de imágenes basado en una red neuronal convolucional (CNN) utilizando TensorFlow y Keras.
El programa está diseñado para clasificar imágenes de autos en cinco categorías específicas: Chevelle, Corolla, Corvette, Murcielago y Quattro. Primero,
carga y procesa las imágenes desde un directorio, ajustando su tamaño y normalizándolas. Las etiquetas de las categorías se transforman a formato one-hot. 
Luego, divide los datos en conjuntos de entrenamiento y prueba (80/20). El modelo de la CNN se construye con varias capas convolucionales, de agrupación 
(pooling) y normalización, seguidas de capas densas para la clasificación final. El modelo se entrena con el optimizador Adam, calculando la pérdida por entropía 
cruzada categórica y evaluando la precisión. Después del entrenamiento, el modelo se guarda en un archivo y se evalúa en el conjunto de prueba. También se genera 
un informe de clasificación y gráficas de pérdida y precisión para visualizar el rendimiento. Por último, se incluye una función para predecir la categoría de nuevas 
imágenes cargando el modelo guardado y mostrando la clase con mayor confianza.

+ Proyecto4:
El objetivo de este proyecto fue entrenar un modelo de inteligencia artificial para analizar y
responder preguntas relacionadas con la reforma al poder judicial y a los organismos autónomos en
México. Esto incluyó un proceso de recopilación, estructuración y entrenamiento del modelo con
información fundamentada en textos legales, académicos y periodísticos.
El análisis se estructuró en dos áreas principales: la reforma al poder judicial y la reforma a los
organismos autónomos. En este reporte, se documentan los algoritmos utilizados, herramientas
generadas, datos empleados, y el proceso de análisis llevado a cabo.
Link del video:
https://www.youtube.com/watch?v=tPna3a4Ij1U
