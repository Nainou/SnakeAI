import pygame
import numpy as np
import random


class FroggerGameRL:
    """
    Simple Frogger-like grid world with cars and trucks:

    - Grid with width x height.
    - Frog moves on a coarse grid (cells).
    - Vehicles (cars & trucks) move on a finer grid horizontally:
      * Each cell is divided into 3 "sub-squares".
      * Cars occupy 3 consecutive sub-squares (≈ 1 full cell in width).
      * Trucks occupy 6 consecutive sub-squares (≈ 2 cells in width).
      * On each step, vehicles move by 1 sub-square -> smoother motion.
      * Collision if ANY of a vehicle's sub-squares overlap with ANY of the frog
        cell's 3 sub-squares.

    - Frog starts at bottom row (start_row = height - 1), in the middle column.
    - Goal is the top row (row 0).
    - Rows 1..height-2 are road lanes with vehicles moving left/right.

    Episode ends when:
        * Frog collides with a vehicle (negative reward).
        * Score reaches target_score (win).
        * Max steps exceeded (small negative).

    Reaching the goal row:
        * Increases score by 1.
        * Respawns frog at start row for another crossing attempt.

    Actions:
        0 = stay
        1 = up
        2 = down
        3 = left
        4 = right
    """

    def __init__(self, grid_width=10, grid_height=12, display=True, render_delay=0):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.display = display
        self.render_delay = render_delay

        # Target score: number of successful crossings to "win" the game
        self.target_score = 10

        # Underlying horizontal sub-grid for vehicles
        # Each cell is split into 3 sub-squares
        self.subcells_per_cell = 3
        self.lane_length_sub = self.grid_width * self.subcells_per_cell

        # Vehicle lengths in subcells
        self.car_length_sub = self.subcells_per_cell          # 1 cell
        self.truck_length_sub = 2 * self.subcells_per_cell   # 2 cells
        self.truck_probability = 0.3  # probability a spawned vehicle is a truck

        # Pygame setup
        if self.display:
            pygame.init()
            self.width = 600
            self.height = 720
            self.square_w = self.width // self.grid_width
            self.square_h = self.height // self.grid_height
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Frogger RL Training")
            pygame.clock = pygame.time.Clock()
            self.clock = pygame.clock

            # Colors
            self.WHITE = (255, 255, 255)
            self.BLACK = (0, 0, 0)
            self.GRAY = (128, 128, 128)
            self.DARK_GRAY = (80, 80, 80)
            self.GREEN = (0, 200, 0)
            self.RED = (200, 0, 0)
            self.BLUE = (0, 0, 200)      # used for goal & trucks
            self.LIGHT_GREEN = (0, 255, 0)
            self.YELLOW = (255, 255, 0)

        # Lane configuration
        # Rows 1..grid_height-2 are traffic lanes.
        self.lane_rows = list(range(1, self.grid_height - 1))
        self.num_lanes = len(self.lane_rows)

        # Per-lane direction: +1 = move right, -1 = move left
        self.lane_directions = {}
        # Per-lane vehicles: dict[row] -> list of vehicle dicts
        # vehicle = {"base": int, "length": int, "type": "car" | "truck"}
        self.cars = {}

        # Game state
        self.frog_x = None  # cell coordinates
        self.frog_y = None
        self.goal_row = 0
        self.start_row = self.grid_height - 1

        self.steps = 0
        self.max_steps = self.grid_width * self.grid_height * 4
        self.done = False
        self.score = 0  # number of successful crossings this episode

        self.reset()

    def reset(self):
        # Initialize frog at bottom middle
        self.frog_x = self.grid_width // 2
        self.frog_y = self.start_row

        # Initialize lanes
        self.lane_directions = {}
        self.cars = {}

        for row in self.lane_rows:
            # Alternate directions per lane
            if (row % 2) == 0:
                direction = 1  # right
            else:
                direction = -1  # left
            self.lane_directions[row] = direction

            # Initialize vehicles in subcell space
            # We'll keep density similar to cars-only version; some will be trucks.
            num_vehicles = max(1, self.grid_width // 6)  # e.g. 2–3 vehicles on 10-wide grid
            spacing_sub = random.randint(7, self.lane_length_sub // num_vehicles)
            lane_vehicles = []
            start_offset_sub = random.randint(0, spacing_sub - 1)

            for i in range(num_vehicles):
                base = (start_offset_sub + i * spacing_sub) % self.lane_length_sub

                # Randomly choose car vs truck
                if random.random() < self.truck_probability:
                    vtype = "truck"
                    length = self.truck_length_sub
                else:
                    vtype = "car"
                    length = self.car_length_sub

                lane_vehicles.append({
                    "base": base,
                    "length": length,
                    "type": vtype
                })

            self.cars[row] = lane_vehicles

        self.steps = 0
        self.done = False
        self.score = 0  # reset score at start of episode

        return self.get_state()

    # ------------- Environment dynamics -------------

    def _frog_subcells_for_cell_x(self, cell_x: int):
        """
        Return the set of subcell indices (in [0, lane_length_sub)) corresponding
        to the given coarse cell x for this lane.

        Each coarse cell corresponds to 3 subcells:
            cell_x -> {3*cell_x, 3*cell_x + 1, 3*cell_x + 2}
        """
        start = cell_x * self.subcells_per_cell
        return {start + i for i in range(self.subcells_per_cell)}

    def _vehicle_subcells_for_base(self, base: int, length_sub: int):
        """
        Return the set of subcell indices (in [0, lane_length_sub)) occupied by a vehicle
        whose "base" (leftmost) subcell index is 'base' and which spans length_sub
        subcells horizontally.
        """
        return {(base + i) % self.lane_length_sub for i in range(length_sub)}

    def _is_car_at(self, cell_x, cell_y):
        """
        Return True if ANY vehicle overlaps the coarse cell (cell_x, cell_y)
        at subcell resolution.

        Overlap if ANY of the vehicle's sub-squares intersect ANY of the 3 sub-squares
        of the cell.
        """
        if cell_y not in self.cars:
            return False

        frog_subcells = self._frog_subcells_for_cell_x(cell_x)

        for vehicle in self.cars[cell_y]:
            car_subcells = self._vehicle_subcells_for_base(
                vehicle["base"],
                vehicle["length"]
            )
            if frog_subcells & car_subcells:
                return True

        return False

    def _move_cars(self):
        """Move all vehicles by 1 subcell in their lane direction, with wrapping."""
        new_cars = {}
        for row in self.lane_rows:
            direction = self.lane_directions[row]
            new_vehicles = []
            for vehicle in self.cars[row]:
                car_spawning = 0
                new_base = vehicle["base"] + direction
                # make sure new vehicles spawn at the edge of the grid
                if direction == 1 and new_base > self.lane_length_sub:
                    car_spawning = 1
                    new_base = -6
                elif direction == -1 and new_base < -6:
                    car_spawning = 1
                    new_base = self.lane_length_sub + 6
                if car_spawning:
                    # reroll car type at new car spawn
                    if random.random() < self.truck_probability:
                        vehicle["type"] = "truck"
                        vehicle["length"] = self.truck_length_sub
                    else:
                        vehicle["type"] = "car"
                        vehicle["length"] = self.car_length_sub 

                new_vehicles.append({
                    "base": new_base,
                    "length": vehicle["length"],
                    "type": vehicle["type"],
                })
            new_cars[row] = new_vehicles
        self.cars = new_cars

    def step(self, action):
        """
        Apply one action:
        action: int in {0..4}
            0 = stay
            1 = up
            2 = down
            3 = left
            4 = right
        """
        if self.done:
            return self.get_state(), 0.0, True, {"score": self.score}

        self.steps += 1
        old_y = self.frog_y

        # ----- 1) Move frog according to action (still on coarse grid) -----
        new_x, new_y = self.frog_x, self.frog_y
        if action == 1:      # up
            new_y -= 1
        elif action == 2:    # down
            new_y += 1
        elif action == 3:    # left
            new_x -= 1
        elif action == 4:    # right
            new_x += 1

        # Clamp to grid (can't move outside)
        new_x = max(0, min(self.grid_width - 1, new_x))
        new_y = max(0, min(self.grid_height - 1, new_y))

        self.frog_x, self.frog_y = new_x, new_y

        reward = 0.0
        info = {}

        # ----- 2) Shaping: encourage moving up, penalize moving down -----
        if self.frog_y < old_y:
            reward += 1.0   # moved closer to goal
        elif self.frog_y > old_y:
            reward -= 1.0   # moved away from goal

        # ----- 3) Goal check (before vehicles move; no traffic on goal row) -----
        if self.frog_y == self.goal_row:
            self.score += 1
            reward += 20.0
            info["reached_goal"] = True

            # Win condition
            if self.score >= self.target_score:
                self.done = True
                reward += 50.0  # bonus for completing the full run
                info["win"] = True
                return self.get_state(), reward, self.done, info

            # Respawn frog at start for next crossing attempt
            self.frog_x = self.grid_width // 2
            self.frog_y = self.start_row

        # Small time penalty to avoid endless wandering
        reward -= 0.01

        # ----- 4) Move vehicles on the subcell grid -----
        self._move_cars()

        # ----- 5) Collision check after vehicles have moved -----
        if self._is_car_at(self.frog_x, self.frog_y):
            self.done = True
            reward = -10.0
            info["crash"] = True
            return self.get_state(), reward, self.done, info

        # ----- 6) Max steps timeout -----
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            reward -= 5.0
            print(f"DEBUG: ran out of time at score {self.score}")
            info["timeout"] = True

        return self.get_state(), reward, self.done, info

    # ------------- State representation -------------

    def get_state(self):
        """
        State vector (unchanged layout):

        - Frog position (normalized): 2 values
        - Vertical direction to goal: 1 value in {-1, 0, 1}
        - Danger in 8 neighboring cells (N, NE, E, SE, S, SW, W, NW): 8 values in {0,1}
        - Distance to nearest vehicle in same column above frog (normalized): 1 value
        - Distance to nearest vehicle in same column below frog (normalized): 1 value
        - Lane direction at frog's row (one-hot: left/right/none): 3 values
        - Relative horizontal position of nearest vehicle in frog's lane (subcell-aware, normalized): 1 value
        - Normalized step count: 1 value

        Total: 2 + 1 + 8 + 1 + 1 + 3 + 1 + 1 = 18
        """
        state = []

        # 1) Frog position normalized
        fx = self.frog_x / (self.grid_width - 1)
        fy = self.frog_y / (self.grid_height - 1)
        state.extend([fx, fy])

        # 2) Vertical direction to goal
        if self.frog_y > self.goal_row:
            goal_dir_y = -1.0
        elif self.frog_y < self.goal_row:
            goal_dir_y = 1.0  # not really used; just for completeness
        else:
            goal_dir_y = 0.0
        state.append(goal_dir_y)

        # 3) Danger in 8 neighboring cells
        neighbor_offsets = [
            (0, -1),   # N
            (1, -1),   # NE
            (1, 0),    # E
            (1, 1),    # SE
            (0, 1),    # S
            (-1, 1),   # SW
            (-1, 0),   # W
            (-1, -1)   # NW
        ]

        for dx, dy in neighbor_offsets:
            nx = self.frog_x + dx
            ny = self.frog_y + dy
            danger = 0
            if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                danger = 1
            elif self._is_car_at(nx, ny):
                danger = 1
            state.append(float(danger))

        # 4) Distance to nearest vehicle in same column above frog (in cell units)
        dist_up = 0.0
        found = False
        for y in range(self.frog_y - 1, -1, -1):
            if self._is_car_at(self.frog_x, y):
                dist_up = (self.frog_y - y) / (self.grid_height - 1)
                found = True
                break
        if not found:
            dist_up = 1.0  # no vehicle above -> treat as far away
        state.append(dist_up)

        # 5) Distance to nearest vehicle in same column below frog (in cell units)
        dist_down = 0.0
        found = False
        for y in range(self.frog_y + 1, self.grid_height):
            if self._is_car_at(self.frog_x, y):
                dist_down = (y - self.frog_y) / (self.grid_height - 1)
                found = True
                break
        if not found:
            dist_down = 1.0
        state.append(dist_down)

        # 6) Lane direction at frog's row (one-hot: left, right, none)
        lane_dir_one_hot = [0.0, 0.0, 0.0]  # [left, right, none]
        if self.frog_y in self.lane_rows:
            d = self.lane_directions[self.frog_y]
            if d == -1:
                lane_dir_one_hot[0] = 1.0
            elif d == 1:
                lane_dir_one_hot[1] = 1.0
        else:
            lane_dir_one_hot[2] = 1.0  # safe row (start or goal)
        state.extend(lane_dir_one_hot)

        # 7) Relative horizontal position of nearest vehicle in frog's lane (subcell-aware)
        rel_car_pos = 0.0
        if self.frog_y in self.lane_rows:
            lane_vehicles = self.cars[self.frog_y]
            if lane_vehicles:
                frog_center = self.frog_x + 0.5
                min_abs_dist = None
                best_rel = 0.0

                for vehicle in lane_vehicles:
                    base = vehicle["base"]
                    length = vehicle["length"]
                    # Vehicle center in "cell units"
                    car_center_cell = (base + length / 2.0) / self.subcells_per_cell
                    d = car_center_cell - frog_center  # positive if vehicle to the right
                    if (min_abs_dist is None) or (abs(d) < min_abs_dist):
                        min_abs_dist = abs(d)
                        best_rel = d

                rel_car_pos = max(-1.0, min(1.0, best_rel / self.grid_width))

        state.append(rel_car_pos)

        # 8) Relative horizontal position of nearest vehicle in next lane (subcell-aware)
        next_car_pos = 0.0
        next_y = self.frog_y - 1
        if next_y in self.lane_rows:
            lane_vehicles = self.cars[next_y]
            if lane_vehicles:
                frog_center = self.frog_x + 0.5
                min_abs_dist = None
                best_rel = 0.0

                for vehicle in lane_vehicles:
                    base = vehicle["base"]
                    length = vehicle["length"]
                    # Vehicle center in "cell units"
                    car_center_cell = (base + length / 2.0) / self.subcells_per_cell
                    d = car_center_cell - frog_center  # positive if vehicle to the right
                    if (min_abs_dist is None) or (abs(d) < min_abs_dist):
                        min_abs_dist = abs(d)
                        best_rel = d

                next_car_pos = max(-1.0, min(1.0, best_rel / self.grid_width))

        state.append(next_car_pos)

        # 9) Relative horizontal position of nearest vehicle in previous lane (subcell-aware)
        prev_car_pos = 0.0
        prev_y = self.frog_y + 1
        if prev_y in self.lane_rows:
            lane_vehicles = self.cars[prev_y]
            if lane_vehicles:
                frog_center = self.frog_x + 0.5
                min_abs_dist = None
                best_rel = 0.0

                for vehicle in lane_vehicles:
                    base = vehicle["base"]
                    length = vehicle["length"]
                    # Vehicle center in "cell units"
                    car_center_cell = (base + length / 2.0) / self.subcells_per_cell
                    d = car_center_cell - frog_center  # positive if vehicle to the right
                    if (min_abs_dist is None) or (abs(d) < min_abs_dist):
                        min_abs_dist = abs(d)
                        best_rel = d

                prev_car_pos = max(-1.0, min(1.0, best_rel / self.grid_width))

        state.append(prev_car_pos)

        # 10) Normalized step count
        step_norm = self.steps / self.max_steps
        state.append(step_norm)

        return np.array(state, dtype=np.float32)

    def get_state_size(self):
        # Length of the state vector
        return len(self.get_state())

    # ------------- Rendering -------------

    def render(self):
        if not self.display:
            return

        self.window.fill(self.BLACK)

        # Draw lanes and safe zones
        for y in range(self.grid_height):
            rect = pygame.Rect(
                0,
                y * self.square_h,
                self.width,
                self.square_h
            )

            if y == self.goal_row:
                color = self.BLUE  # goal
            elif y == self.start_row:
                color = self.LIGHT_GREEN  # start
            elif y in self.lane_rows:
                color = self.DARK_GRAY  # road lanes
            else:
                color = self.GRAY
            pygame.draw.rect(self.window, color, rect)

        # Draw vehicles (visually thinner vertically, with yellow stripe on movement side)
        car_margin = int(self.square_h * 0.15)      # shrink from top and bottom (visual only)
        stripe_width = max(2, self.square_w // 8)   # width of the yellow direction stripe

        for row in self.lane_rows:
            lane_dir = self.lane_directions[row]    # +1 = right, -1 = left
            for vehicle in self.cars[row]:
                base = vehicle["base"]
                vtype = vehicle["type"]

                # Convert base subcell index to a float cell position
                cell_pos_float = base / self.subcells_per_cell
                pixel_x = int(cell_pos_float * self.square_w)

                # Width: cars = 1 cell, trucks = 2 cells
                if vtype == "truck":
                    rect_width = 2 * self.square_w
                    color = self.BLUE
                else:
                    rect_width = self.square_w
                    color = self.RED

                # Base vehicle rectangle, thinner vertically
                car_rect = pygame.Rect(
                    pixel_x,
                    row * self.square_h + car_margin,
                    rect_width,
                    self.square_h - 2 * car_margin
                )
                pygame.draw.rect(self.window, color, car_rect)

                # Yellow stripe indicating direction of movement (full height of the vehicle)
                if lane_dir == 1:
                    # moving right → stripe on the right edge
                    stripe_rect = pygame.Rect(
                        car_rect.right - stripe_width,
                        car_rect.top,
                        stripe_width,
                        car_rect.height
                    )
                else:
                    # moving left → stripe on the left edge
                    stripe_rect = pygame.Rect(
                        car_rect.left,
                        car_rect.top,
                        stripe_width,
                        car_rect.height
                    )

                pygame.draw.rect(self.window, self.YELLOW, stripe_rect)

        # Draw frog (coarse grid)
        frog_rect = pygame.Rect(
            self.frog_x * self.square_w + self.square_w // 4,
            self.frog_y * self.square_h + self.square_h // 4,
            self.square_w // 2,
            self.square_h // 2
        )
        pygame.draw.rect(self.window, self.GREEN, frog_rect)

        # Draw text (score and steps)
        font = pygame.font.Font(None, 32)
        text_surface = font.render(
            f"Score: {self.score}/{self.target_score}  Steps: {self.steps}",
            True,
            self.WHITE
        )
        self.window.blit(text_surface, (10, 10))

        pygame.display.flip()

        if self.render_delay > 0:
            self.clock.tick(self.render_delay)

    def close(self):
        if self.display:
            pygame.quit()


if __name__ == "__main__":
    # Human play test
    game = FroggerGameRL(display=True, render_delay=10)

    print("Controls: arrow keys to move, SPACE to stay, ESC to quit.")
    print(f"State size: {game.get_state_size()}")

    running = True
    state = game.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_RIGHT:
                    action = 4
                elif event.key == pygame.K_SPACE:
                    action = 0

                if action is not None:
                    state, reward, done, info = game.step(action)
                    print("Reward:", reward, "Done:", done, "Info:", info)
                    if done:
                        if info.get("win"):
                            print(f"Episode finished with a WIN! Score = {game.score}")
                        elif info.get("crash"):
                            print(f"Episode finished by CRASH. Score = {game.score}")
                        elif info.get("timeout"):
                            print(f"Episode finished by TIMEOUT. Score = {game.score}")
                        else:
                            print(f"Episode finished. Score = {game.score}")
                        state = game.reset()

        game.render()

    game.close()
