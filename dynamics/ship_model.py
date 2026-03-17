import numpy as np


class ShipModel:

    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=1.0, map_size=100):

        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)

        self.speed        = speed
        self.target_speed = speed

        self.yaw_rate = 0.0

        self.map_size = map_size

        # ── Tuned Physical parameters ─────────────────────────────

        # balanced steering
        self.turn_gain = 0.75

        # stronger damping prevents oscillation
        self.turn_damping = 0.95

        # maximum turning rate
        self.max_yaw_rate = 0.65

        # smoother throttle
        self.acceleration = 0.65

        # slightly slower top speed improves avoidance
        self.max_speed = 4.2
        self.min_speed = 0.25

    # ------------------------------------------------
    # Reset
    # ------------------------------------------------
    def reset(self, x, y, heading):

        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)

        self.speed        = 1.0
        self.target_speed = 1.0
        self.yaw_rate     = 0.0

    # ------------------------------------------------
    # Step dynamics
    # ------------------------------------------------
    def step(self, rudder, throttle, dt):

        rudder   = float(np.clip(rudder, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # ── Speed control ───────────────────────────

        self.target_speed = self.min_speed + throttle * (self.max_speed - self.min_speed)

        speed_error  = self.target_speed - self.speed
        self.speed  += speed_error * self.acceleration * dt
        self.speed   = np.clip(self.speed, self.min_speed, self.max_speed)

        # ── Turning dynamics ───────────────────────

        speed_factor = self.speed / self.max_speed

        yaw_acc = rudder * self.turn_gain * speed_factor

        self.yaw_rate += yaw_acc * dt
        self.yaw_rate *= self.turn_damping
        self.yaw_rate  = np.clip(self.yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # ── Reduce speed during sharp turns (prevents overshoot)

        turn_factor = abs(self.yaw_rate) / self.max_yaw_rate
        speed_limit = self.max_speed * (1 - 0.5 * turn_factor)

        self.speed = min(self.speed, speed_limit)

        # ── Update heading

        self.heading += self.yaw_rate * dt
        self.heading  = np.arctan2(np.sin(self.heading), np.cos(self.heading))

        # ── Position update

        self.x += self.speed * np.cos(self.heading) * dt
        self.y += self.speed * np.sin(self.heading) * dt

        # ── Boundary clamp (prevents leaving map)

        margin = 1.0

        self.x = np.clip(self.x, margin, self.map_size - margin)
        self.y = np.clip(self.y, margin, self.map_size - margin)

        return self.get_state()

    # ------------------------------------------------
    # Return state
    # ------------------------------------------------
    def get_state(self):

        return np.array(
            [self.x, self.y, self.heading, self.speed, self.yaw_rate],
            dtype=np.float32
        )