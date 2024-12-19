import pygame
import random
import heapq

WIDTH, HEIGHT = 500, 500
ROWS, COLS = 5, 5
CELL_SIZE = WIDTH // COLS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Inicia pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("A* - Laberinto chido")
clock = pygame.time.Clock()

class Node:
    def __init__(self, x, y, walkable=True):
        self.x = x
        self.y = y
        self.walkable = walkable
        self.g = float('inf')
        self.f = float('inf')
        self.neighbors = []
        self.previous = None
        self.explored = False

    def __lt__(self, other):
        return self.f < other.f

    def draw(self, color):
        pygame.draw.rect(screen, color, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, BLACK, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

# Crear laberinto vacío
def generate_empty_maze():
    maze = [[Node(x, y, walkable=True) for y in range(ROWS)] for x in range(COLS)]
    return maze

# Aquí se conectan los vecinos
def connect_neighbors(grid):
    for x in range(COLS):
        for y in range(ROWS):
            node = grid[x][y]
            if node.walkable:
                # verticales y horizontales
                if x > 0 and grid[x - 1][y].walkable:
                    node.neighbors.append(grid[x - 1][y])
                if x < COLS - 1 and grid[x + 1][y].walkable:
                    node.neighbors.append(grid[x + 1][y])
                if y > 0 and grid[x][y - 1].walkable:
                    node.neighbors.append(grid[x][y - 1])
                if y < ROWS - 1 and grid[x][y + 1].walkable:
                    node.neighbors.append(grid[x][y + 1])

                # diagonales
                if x > 0 and y > 0 and grid[x - 1][y - 1].walkable:
                    node.neighbors.append(grid[x - 1][y - 1])
                if x < COLS - 1 and y > 0 and grid[x + 1][y - 1].walkable:
                    node.neighbors.append(grid[x + 1][y - 1])
                if x > 0 and y < ROWS - 1 and grid[x - 1][y + 1].walkable:
                    node.neighbors.append(grid[x - 1][y + 1])
                if x < COLS - 1 and y < ROWS - 1 and grid[x + 1][y + 1].walkable:
                    node.neighbors.append(grid[x + 1][y + 1])

# Heurística
def heuristic(node1, node2):
    return max(abs(node1.x - node2.x), abs(node1.y - node2.y))

# Algoritmo a* con diagonales
def a_star(start, end, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    start.g = 0
    start.f = heuristic(start, end)

    path = []  # Almacenar el camino más corto

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        current = heapq.heappop(open_set)[1]

        # Marca nodo como "explorado"
        current.explored = True

        if current == end:
            path = reconstruct_path(end)
            break

        for neighbor in current.neighbors:
            tentative_g = current.g + (1 if neighbor.x == current.x or neighbor.y == current.y else 1.414)  # 1 para vertical/horizontal, ~1.414 para diagonal
            if tentative_g < neighbor.g:
                neighbor.previous = current
                neighbor.g = tentative_g
                neighbor.f = tentative_g + heuristic(neighbor, end)
                if not any(neighbor == item[1] for item in open_set):
                    heapq.heappush(open_set, (neighbor.f, neighbor))

        # Visualización
        screen.fill(WHITE)
        draw_grid(grid)
        for row in grid:
            for node in row:
                if node.explored and node != start and node != end:
                    node.draw(YELLOW)
        current.draw(GREEN)
        start.draw(RED)
        end.draw(RED)
        pygame.display.update()
        clock.tick(10)

    return path

# Reconstruir el camino
def reconstruct_path(end):
    path = []
    current = end
    while current.previous:
        path.append(current)
        current = current.previous
    path.reverse()
    for node in path:
        node.draw(BLUE)  # Resalta la ruta + corta
    pygame.display.update()
    return path

# Cuadrícula
def draw_grid(grid):
    for row in grid:
        for node in row:
            color = WHITE if node.walkable else GRAY
            node.draw(color)

# Programa principal
def main():
    grid = generate_empty_maze()

    start = grid[0][0]  # NodoInicial
    end = grid[COLS - 1][ROWS - 1]  # NodoFinal

    running = True
    drawing = True
    solving = False
    path = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and drawing:
                x, y = pygame.mouse.get_pos()
                clicked_node = grid[x // CELL_SIZE][y // CELL_SIZE]
                if pygame.mouse.get_pressed()[2]:  # Clic derecho para inicio
                    start = clicked_node
                elif pygame.mouse.get_pressed()[1]:  # Clic central para fin
                    end = clicked_node
                else:  # Clic izquierdo para paredes
                    clicked_node.walkable = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if drawing:
                        drawing = False
                        connect_neighbors(grid)
                    elif not solving:
                        solving = True
                        path = a_star(start, end, grid)

        screen.fill(WHITE)
        draw_grid(grid)
        start.draw(RED)
        end.draw(RED)

        # Mostrar el camino
        for node in path:
            node.draw(BLUE)

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
