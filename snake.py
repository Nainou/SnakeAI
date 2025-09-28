import pygame
import sys
import random

class Snake:
    def __init__(self):
        self.size = 20
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

grid_size = 10
square_size = WIDTH // grid_size
segment_margin = 5
segment_width = square_size - 2 * segment_margin
overlap = 2 * segment_margin
segment_length = square_size + overlap
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
            prev = snake.positions[i - 1] if i > 0 else None
            next = snake.positions[i + 1] if i < len(snake.positions) - 1 else None

            x = position[0] * square_size
            y = position[1] * square_size

            if i == 0:
                color = BLACK
                segment_margin = 5
                segment_width = square_size - 2 * segment_margin
                overlap = 2 * segment_margin - 1
                segment_length = square_size + overlap
            else:
                color = (0, 120, 0)
                segment_margin = 5
                segment_width = square_size - 2 * segment_margin
                overlap = 2 * segment_margin
                segment_length = square_size + overlap

            #get direction
            def get_dir(a, b):
                if not a or not b:
                    return None
                return (b[0] - a[0], b[1] - a[1])

            dir_from_prev = get_dir(prev, position)
            dir_to_next = get_dir(position, next)

            #detect corners
            def is_perpendicular(a, b):
                return a and b and a != b and (a[0]*b[0] + a[1]*b[1] == 0)

            is_corner = is_perpendicular(dir_from_prev, dir_to_next)

            #detect if previous or next is also a corner
            prev_is_corner = False
            next_is_corner = False
            if i > 1:
                prev_prev = snake.positions[i - 2]
                dir_prev_prev = get_dir(prev_prev, prev)
                prev_is_corner = is_perpendicular(dir_prev_prev, dir_from_prev)
            if i < len(snake.positions) - 2:
                next_next = snake.positions[i + 2]
                dir_next_next = get_dir(next, next_next)
                next_is_corner = is_perpendicular(dir_to_next, dir_next_next)

            if is_corner:
                base_x = x + segment_margin
                base_y = y + segment_margin
                base_w = segment_width
                base_h = segment_width

                headward = (-dir_from_prev[0], -dir_from_prev[1]) if dir_from_prev else None
                tailward = dir_to_next if dir_to_next else None

                for d in [headward, tailward]:
                    if d[0] != 0:
                        sign = 1 if d[0] > 0 else -1
                        # Use full ext if two corners in a row, else reduced
                        if (prev_is_corner and d == headward) or (next_is_corner and d == tailward):
                            ext = segment_margin
                        else:
                            ext = segment_margin - 5
                        base_w += ext
                        if sign < 0:
                            base_x -= ext
                    if d[1] != 0:
                        sign = 1 if d[1] > 0 else -1
                        if (prev_is_corner and d == headward) or (next_is_corner and d == tailward):
                            ext = segment_margin
                        else:
                            ext = segment_margin - 5
                        base_h += ext
                        if sign < 0:
                            base_y -= ext

                rect = pygame.Rect(base_x, base_y, base_w, base_h)
                pygame.draw.rect(WINDOW, color, rect)
            else:
                # Straight segment
                if dir_from_prev is None and dir_to_next is None:
                    # Single segment
                    rect = pygame.Rect(x + segment_margin, y + segment_margin, segment_width, segment_width)
                elif (dir_from_prev and dir_from_prev[0] != 0) or (dir_to_next and dir_to_next[0] != 0):
                    # Horizontal
                    rect = pygame.Rect(x - segment_margin, y + segment_margin, segment_length, segment_width)
                elif (dir_from_prev and dir_from_prev[1] != 0) or (dir_to_next and dir_to_next[1] != 0):
                    # Vertical
                    rect = pygame.Rect(x + segment_margin, y - segment_margin, segment_width, segment_length)
                else:
                    # Fallback
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
            pygame.draw.rect(WINDOW, tongue_color, (hx + square_size, hy + square_size//2 - tongue_width//2, tongue_length, tongue_width))
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