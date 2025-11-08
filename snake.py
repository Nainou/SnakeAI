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
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (170, 215, 81)
DARK_GREEN = (162, 209, 73)

#Score counter
score_font = pygame.font.SysFont("None", 36)

grid_size = 10
square_size = WIDTH // grid_size
segment_margin = 5
food_margin = 7
segment_width = square_size - 2 * segment_margin
print("grid_size:", grid_size, "square_size:", square_size,"pixels")

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
        rect = pygame.Rect(food.position[0]*square_size + food_margin,food.position[1]*square_size + food_margin,square_size - food_margin*2 ,square_size - food_margin*2)
        pygame.draw.rect(WINDOW, RED, rect)

        #draw snake
        for i, position in enumerate(snake.positions):
            x = position[0] * square_size
            y = position[1] * square_size
            prev = snake.positions[i - 1] if i > 0 else None

            # Draw base square segment
            if i == 0:
                color = (0, 120, 0)
                segment_margin = 5
            else:
                color = (0, 120, 0)
                segment_margin = 5

                # Draw connection to previous segment
                if prev:
                    # Calculate direction to previous segment
                    dx = prev[0] - position[0]
                    dy = prev[1] - position[1]

                    # Handle screen wrap-around
                    if abs(dx) > 1: # Wrapped horizontally
                        dx = -1 if dx > 0 else 1
                    if abs(dy) > 1: # Wrapped vertically
                        dy = -1 if dy > 0 else 1

                    # Draw connecting rectangle
                    if dx != 0:  # Horizontal connection
                        connect_x = x + segment_margin
                        if dx < 0:  # Going left
                            connect_x = x - (square_size - segment_margin * 2)
                        if dx > 0:  # Going right
                            connect_x = x + (square_size - segment_margin * 2)
                        connect_y = y + segment_margin
                        connect_w = square_size - segment_margin * 2 + 5
                        connect_h = segment_width
                        pygame.draw.rect(WINDOW, color, (connect_x, connect_y, connect_w, connect_h))
                    elif dy != 0:  # Vertical connection
                        connect_x = x + segment_margin
                        connect_y = y + segment_margin
                        if dy < 0:  # Going up
                            connect_y = y - (square_size - segment_margin * 2)
                        if dy > 0:  # Going down
                            connect_y = y + (square_size - segment_margin * 2)
                        connect_w = segment_width
                        connect_h = square_size - segment_margin * 2 + 5
                        pygame.draw.rect(WINDOW, color, (connect_x, connect_y, connect_w, connect_h))

            # Draw segment square
            rect = pygame.Rect(x + segment_margin, y + segment_margin, segment_width, segment_width)
            pygame.draw.rect(WINDOW, color, rect)

        # Draw tongue
        hx, hy = snake.head[0]*square_size, snake.head[1]*square_size
        dx, dy = snake.direction
        tongue_color = (255,0,0)
        tongue_length = 8
        tongue_width = 2
        fork_size = 2
        if dx == 1:  # Right
            pygame.draw.rect(WINDOW, tongue_color, (hx - 5 + square_size, hy + square_size//2 - tongue_width//2, tongue_length, tongue_width))
            pygame.draw.rect(WINDOW, tongue_color, (hx - 5 + square_size + tongue_length, hy + square_size//2 - tongue_width//2 - fork_size, fork_size, fork_size))
            pygame.draw.rect(WINDOW, tongue_color, (hx - 5 + square_size + tongue_length, hy + square_size//2 - tongue_width//2 + tongue_width, fork_size, fork_size))
        elif dx == -1:  # Left
            pygame.draw.rect(WINDOW, tongue_color, (hx + 5 - tongue_length, hy + square_size//2 - tongue_width//2, tongue_length, tongue_width))
            pygame.draw.rect(WINDOW, tongue_color, (hx + 5 - tongue_length - fork_size, hy + square_size//2 - tongue_width//2 - fork_size, fork_size, fork_size))
            pygame.draw.rect(WINDOW, tongue_color, (hx + 5 - tongue_length - fork_size, hy + square_size//2 - tongue_width//2 + tongue_width, fork_size, fork_size))
        elif dy == 1:  # Down
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2, hy - 5 + square_size, tongue_width, tongue_length))
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 - fork_size, hy - 5 + square_size + tongue_length, fork_size, fork_size))
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 + tongue_width, hy - 5 + square_size + tongue_length, fork_size, fork_size))
        else:  # Up
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2, hy + 5 - tongue_length, tongue_width, tongue_length))
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 - fork_size, hy + 5 - tongue_length - fork_size, fork_size, fork_size))
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size//2 - tongue_width//2 + tongue_width, hy + 5 - tongue_length - fork_size, fork_size, fork_size))

        # Draw eyes
        eye_color = BLACK
        eye_size = 4
        if dx == 1:  # Right
            pygame.draw.rect(WINDOW, eye_color, (hx + 5 + square_size//2, hy + segment_margin + 5, eye_size, eye_size))
            pygame.draw.rect(WINDOW, eye_color, (hx + 5 + square_size//2, hy + square_size - segment_margin - 5 - eye_size, eye_size, eye_size))
        elif dx == -1:  # Left
            pygame.draw.rect(WINDOW, eye_color, (hx + 5 + square_size//2 - eye_size, hy + segment_margin + 5, eye_size, eye_size))
            pygame.draw.rect(WINDOW, eye_color, (hx + 5 + square_size//2 - eye_size, hy + square_size - segment_margin - 5 - eye_size, eye_size, eye_size))
        elif dy == 1:  # Down
            pygame.draw.rect(WINDOW, eye_color, (hx + segment_margin + 5, hy + 5 + square_size//2, eye_size, eye_size))
            pygame.draw.rect(WINDOW, eye_color, (hx + square_size - segment_margin - 5 - eye_size, hy + 5 + square_size//2, eye_size, eye_size))
        else:  # Up
            pygame.draw.rect(WINDOW, eye_color, (hx + segment_margin + 5, hy + 5 + square_size//2 - eye_size, eye_size, eye_size))
            pygame.draw.rect(WINDOW, eye_color, (hx + square_size - segment_margin - 5 - eye_size, hy + 5 + square_size//2 - eye_size, eye_size, eye_size))

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