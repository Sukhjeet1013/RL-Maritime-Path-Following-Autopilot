import numpy as np


class OceanCurrent:

    def __init__(self, base_strength=0.05, noise_std=0.005):
        """
        Ocean current disturbance model
        (weakened for stable RL training)
        """

        self.base_strength = base_strength
        self.noise_std = noise_std

        # slowly varying turbulence bias
        self.bias = np.zeros(2)

        # maximum current magnitude
        self.max_current = 0.15

    # --------------------------------------------------
    # Reset turbulence each episode
    # --------------------------------------------------
    def reset(self):

        self.bias = np.zeros(2)

    # --------------------------------------------------
    # Spatial ocean current field
    # --------------------------------------------------
    def current_field(self, x, y):

        """
        Smooth spatial flow field
        """

        cx = self.base_strength * (0.5 + 0.005 * x)

        cy = 0.05 * np.sin(y / 20)

        return np.array([cx, cy])

    # --------------------------------------------------
    # Turbulence
    # --------------------------------------------------
    def turbulence(self):

        noise = np.random.normal(0, self.noise_std, size=2)

        # temporal smoothing
        self.bias = 0.9 * self.bias + noise

        # limit turbulence
        self.bias = np.clip(self.bias, -0.02, 0.02)

        return self.bias

    # --------------------------------------------------
    # Total current
    # --------------------------------------------------
    def get_current(self, x, y):

        field = self.current_field(x, y)

        noise = self.turbulence()

        current = field + noise

        # limit total current magnitude
        mag = np.linalg.norm(current)

        if mag > self.max_current:
            current = current / mag * self.max_current

        return current

    # --------------------------------------------------
    # Apply disturbance
    # --------------------------------------------------
    def apply_current(self, x, y, dt):

        current = self.get_current(x, y)

        x += current[0] * dt
        y += current[1] * dt

        return x, y, current