import numpy as np


class ObstacleField:

    def __init__(self, num_obstacles=3, map_size=100, radius=2.0):

        self.num_obstacles = num_obstacles
        self.map_size = map_size
        self.radius = radius

        self.obstacles = []

        # expanded exclusion zones so obstacles don't block start→goal path
        self.exclusion_zones = [
            (10, 10),
            (30, 20),
            (60, 40),
            (90, 80)
        ]

        self.reset()

    # ------------------------------------------------
    # Reset obstacle positions
    # ------------------------------------------------
    def reset(self):

        self.obstacles = []

        for _ in range(self.num_obstacles):

            while True:

                x = np.random.uniform(15, self.map_size - 15)
                y = np.random.uniform(15, self.map_size - 15)

                valid = True

                for zx, zy in self.exclusion_zones:

                    dist = np.sqrt((x - zx)**2 + (y - zy)**2)

                    # bigger safe zone
                    if dist < 15:
                        valid = False
                        break

                if valid:
                    break

            # slower obstacle velocity
            vx = np.random.uniform(-0.1, 0.1)
            vy = np.random.uniform(-0.1, 0.1)

            self.obstacles.append([x, y, vx, vy])

    # ------------------------------------------------
    # Set obstacle count
    # ------------------------------------------------
    def set_num_obstacles(self, n):

        self.num_obstacles = n
        self.reset()

    # ------------------------------------------------
    # Move obstacles
    # ------------------------------------------------
    def step(self):

        for obs in self.obstacles:

            x, y, vx, vy = obs

            # smaller randomness → smoother motion
            vx += np.random.uniform(-0.005, 0.005)
            vy += np.random.uniform(-0.005, 0.005)

            # lower max speed
            vx = np.clip(vx, -0.15, 0.15)
            vy = np.clip(vy, -0.15, 0.15)

            x += vx
            y += vy

            if x < self.radius or x > self.map_size - self.radius:
                vx *= -1
                x = np.clip(x, self.radius, self.map_size - self.radius)

            if y < self.radius or y > self.map_size - self.radius:
                vy *= -1
                y = np.clip(y, self.radius, self.map_size - self.radius)

            obs[0] = x
            obs[1] = y
            obs[2] = vx
            obs[3] = vy

    # ------------------------------------------------
    # Get obstacle positions
    # ------------------------------------------------
    def get_positions(self):
        return [(obs[0], obs[1]) for obs in self.obstacles]

    # ------------------------------------------------
    # Get obstacle velocities
    # ------------------------------------------------
    def get_velocities(self):
        return [(obs[2], obs[3]) for obs in self.obstacles]

    # ------------------------------------------------
    # Get full obstacle info
    # ------------------------------------------------
    def get_all_obstacle_info(self):
        return [(obs[0], obs[1], obs[2], obs[3]) for obs in self.obstacles]

    # ------------------------------------------------
    # Get obstacle radius
    # ------------------------------------------------
    def get_radius(self):
        return self.radius