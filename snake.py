import pygame
import sys
import random

class Snake:
    def __init__(self):
        self.size = 1
        self.positions = [(3, 4)] #starting pos
        self.head = self.positions[0]
        self.tail = self.positions[-1]
        self.direction=(1,0) #move right
        self.turning = False

    def move(self):
        #move head one square in the current direction
        self.head = (self.head[0]+self.direction[0],self.head[1]+self.direction[1])

        #if head goes out of bounds on x-axis
        if self.head[0] >= grid_size:
            self.head = (0,self.head[1])
        if self.head[0] < 0:
            self.head = (grid_size-1,self.head[1])

        #if head goes out of bounds on y-axis
        if self.head[1] >= grid_size:
            self.head = (self.head[0],0)
        if self.head[1] < 0:
            self.head = (self.head[0],grid_size-1)

        #add new head
        self.positions.insert(0,self.head)

        #remove tail
        if len(self.positions) > self.size:
           self.positions.pop()

        self.turning = False

class Food:
    def __init__(self, x, y):
        self.position = (x, y)
    def respawn(self,snake_positions):
        while True:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            if (x,y) not in snake_positions:
                self.position = (x, y)
                break

pygame.init()

WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (170, 215, 81)
DARK_GREEN = (162, 209, 73)

#Score counter
score_font = pygame.font.SysFont("None", 36)

grid_size = 10 # y and x grid size
square_size = WIDTH // grid_size
print("grid_size:", grid_size, "square_size:", square_size,"pixels")
grid = []

clock = pygame.time.Clock()
FPS = 6

snake = Snake()
food = Food(random.randint(0, grid_size-1), random.randint(0, grid_size-1))

def main():
    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN and not snake.turning:
                if event.key == pygame.K_UP and snake.direction != (0,1):
                    snake.direction=(0,-1)
                elif event.key==pygame.K_DOWN and snake.direction != (0,-1):
                    snake.direction=(0,1)
                elif event.key==pygame.K_LEFT and snake.direction != (1,0):
                    snake.direction=(-1,0)
                elif event.key==pygame.K_RIGHT and snake.direction != (-1,0):
                    snake.direction=(1,0)
                snake.turning = True

        snake.move()

        if snake.head in snake.positions[1:]:
            running = False

        if snake.head == food.position:
            snake.size += 1
            food.respawn(snake.positions)

        for x in range(grid_size):
            for y in range(grid_size):
                color = LIGHT_GREEN if (x + y) % 2 == 0 else DARK_GREEN
                pygame.draw.rect(WINDOW, color, (x * square_size, y * square_size, square_size, square_size))

        #draw food
        rect = pygame.Rect(food.position[0]*square_size,food.position[1]*square_size,square_size,square_size)
        pygame.draw.rect(WINDOW, RED, rect)

        #draw snake
        for i, position in enumerate(snake.positions):
            x = position[0] * square_size
            y = position[1] * square_size
            if position == snake.head:
                rect = pygame.Rect(position[0]*square_size,position[1]*square_size,square_size,square_size)
                pygame.draw.rect(WINDOW, BLACK, rect)

                hx, hy = position[0]*square_size, position[1]*square_size
                dx, dy = snake.direction
                tongue_color = (255,0,0)
                # Tongue parameters
                tongue_length = 8
                tongue_width = 2
                fork_size = 2
                if dx == 1:  # Right
                    # Straight part
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size, hy + square_size//2 - tongue_width//2, tongue_length, tongue_width))
                    # Split ends
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size + tongue_length, hy + square_size//2 - tongue_width//2 - fork_size, fork_size, fork_size))
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size + tongue_length, hy + square_size//2 - tongue_width//2 + tongue_width, fork_size, fork_size))
                elif dx == -1:  # Left
                    pygame.draw.rect(WINDOW, tongue_color, (hx - tongue_length, hy + square_size//2 - tongue_width//2, tongue_length, tongue_width))
                    pygame.draw.rect(WINDOW, tongue_color, (hx - tongue_length - fork_size, hy + square_size//2 - tongue_width//2 - fork_size, fork_size, fork_size))
                    pygame.draw.rect(WINDOW, tongue_color, (hx - tongue_length - fork_size, hy + square_size//2 - tongue_width//2 + tongue_width, fork_size, fork_size))
                elif dy == 1:  # Down
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2, hy + square_size, tongue_width, tongue_length))
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 - fork_size, hy + square_size + tongue_length, fork_size, fork_size))
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 + tongue_width, hy + square_size + tongue_length, fork_size, fork_size))
                else:  # Up
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2, hy - tongue_length, tongue_width, tongue_length))
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 - fork_size, hy - tongue_length - fork_size, fork_size, fork_size))
                    pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 + tongue_width, hy - tongue_length - fork_size, fork_size, fork_size))

            else:
                rect = pygame.Rect(position[0]*square_size,position[1]*square_size,square_size,square_size)
                pygame.draw.rect(WINDOW, (0, 120, 0), rect)

        #Score counter
        score_text = score_font.render(f"Score: {snake.size - 1}", True, (0, 0, 0))
        score_bg = pygame.Surface((score_text.get_width() + 16, score_text.get_height() + 8), pygame.SRCALPHA)
        score_bg.fill((0, 0, 0, 100))
        WINDOW.blit(score_bg, (6, 6))
        WINDOW.blit(score_text, (14, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
