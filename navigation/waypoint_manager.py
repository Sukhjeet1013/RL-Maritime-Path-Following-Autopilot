import numpy as np


class WaypointManager:

    def __init__(self, waypoints, threshold=3.0):
        """
        waypoints : list of (x, y)
        threshold : distance to consider waypoint reached
        """

        self.waypoints = waypoints
        self.threshold = threshold
        self.current_index = 0

    # ----------------------------
    # Reset navigation
    # ----------------------------
    def reset(self):

        self.current_index = 0

    # ----------------------------
    # Current waypoint
    # ----------------------------
    def get_current_waypoint(self):

        idx = min(self.current_index, len(self.waypoints) - 1)

        return self.waypoints[idx]

    # ----------------------------
    # Previous waypoint
    # ----------------------------
    def get_previous_waypoint(self):

        # avoid zero-length path segment
        if self.current_index <= 0:
            return self.waypoints[0]

        return self.waypoints[self.current_index - 1]

    # ----------------------------
    # Distance utility
    # ----------------------------
    def _distance(self, x, y, wx, wy):

        dx = wx - x
        dy = wy - y

        return np.sqrt(dx * dx + dy * dy)

    # ----------------------------
    # Distance to current waypoint
    # ----------------------------
    def distance_to_waypoint(self, x, y):

        wx, wy = self.get_current_waypoint()

        return self._distance(x, y, wx, wy)

    # ----------------------------
    # Check waypoint reached
    # ----------------------------
    def check_waypoint_reached(self, x, y):

        if self.current_index >= len(self.waypoints) - 1:
            return False

        wx, wy = self.get_current_waypoint()

        distance = self._distance(x, y, wx, wy)

        # waypoint reached
        if distance < self.threshold:

            self.current_index += 1
            return True

        # safety: detect if ship passed waypoint
        prev_wp = self.get_previous_waypoint()

        vec_path = np.array(wx) - np.array(prev_wp)
        vec_ship = np.array([x, y]) - np.array(prev_wp)

        if np.dot(vec_ship, vec_path) > np.dot(vec_path, vec_path):

            self.current_index += 1
            return True

        return False

    # ----------------------------
    # Final waypoint check
    # ----------------------------
    def is_final_waypoint(self):

        return self.current_index >= len(self.waypoints) - 1

    # ----------------------------
    # Navigation finished
    # ----------------------------
    def is_navigation_complete(self, x, y):

        if not self.is_final_waypoint():
            return False

        wx, wy = self.get_current_waypoint()

        distance = self._distance(x, y, wx, wy)

        return distance < self.threshold