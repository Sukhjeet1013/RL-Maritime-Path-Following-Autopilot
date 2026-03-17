import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dynamics.ship_model import ShipModel
from dynamics.ocean_current import OceanCurrent
from dynamics.obstacle_field import ObstacleField

from navigation.waypoint_manager import WaypointManager
from navigation.path_utils import cross_track_error


class MaritimePathEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.map_size  = 100
        self.dt        = 0.1
        self.max_steps = 2000

        self.ship      = ShipModel()
        self.ocean     = OceanCurrent()
        self.obstacles = ObstacleField()

        self.waypoints = [
            (10, 10),
            (30, 20),
            (60, 40),
            (90, 80)
        ]

        self.waypoint_manager = WaypointManager(self.waypoints)

        self.prev_progress = 0
        self.step_count    = 0

        self.lookahead = 8.0

        # ── Penalty zone radii ─────────────────────────────────────────
        # Tightened from 10/5 → 8/4.
        # The old warn radius of 10 covered 10% of the map — the agent
        # was almost always inside the warn zone and bled -8 to -13 per
        # step continuously, making "stand still" better than "move".
        self.obs_warn_radius   = 8.0
        self.obs_danger_radius = 4.0

        # max relative speed for normalising obstacle velocity obs
        self._max_rel_speed = 3.5

        self.action_space = spaces.Box(
            low  = np.array([-1.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        _INF = self.map_size * 1.5

        # 18-feature observation (unchanged from previous version)
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,        # 0  x
                0.0,        # 1  y
                -np.pi,     # 2  heading
                0.0,        # 3  distance_to_wp
                -np.pi,     # 4  angle_to_wp
                -_INF,      # 5  cte
                -5.0,       # 6  current_x
                -5.0,       # 7  current_y
                0.0,        # 8  speed
                0.0,        # 9  dist_1
                -1.0,       # 10 dx_1
                -1.0,       # 11 dy_1
                -1.0,       # 12 vx_rel_1
                -1.0,       # 13 vy_rel_1
                0.0,        # 14 dist_2
                -1.0,       # 15 dx_2
                -1.0,       # 16 dy_2
                -1.0,       # 17 closing_speed_2
            ], dtype=np.float32),
            high=np.array([
                self.map_size,
                self.map_size,
                np.pi,
                _INF,
                np.pi,
                _INF,
                5.0,
                5.0,
                3.0,
                _INF,
                1.0,
                1.0,
                1.0,
                1.0,
                _INF,
                1.0,
                1.0,
                1.0,
            ], dtype=np.float32),
            dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Obstacle sensing                                                     #
    # ------------------------------------------------------------------ #
    def _obstacle_obs(self, ship_x, ship_y, ship_vx, ship_vy):
        all_obs = self.obstacles.get_all_obstacle_info()

        entries = []
        for ox, oy, ovx, ovy in all_obs:
            ddx  = ox - ship_x
            ddy  = oy - ship_y
            dist = float(np.sqrt(ddx**2 + ddy**2))

            if dist > 1e-6:
                ux = ddx / dist
                uy = ddy / dist
            else:
                ux, uy = 0.0, 1.0

            rvx = float(ovx - ship_vx)
            rvy = float(ovy - ship_vy)
            entries.append((dist, ux, uy, rvx, rvy))

        entries.sort(key=lambda e: e[0])

        d1, ux1, uy1, rvx1, rvy1 = entries[0]
        vx_rel_1 = float(np.clip(rvx1 / self._max_rel_speed, -1.0, 1.0))
        vy_rel_1 = float(np.clip(rvy1 / self._max_rel_speed, -1.0, 1.0))

        if len(entries) >= 2:
            d2, ux2, uy2, rvx2, rvy2 = entries[1]
            closing2 = float(np.clip(
                -(rvx2 * ux2 + rvy2 * uy2) / self._max_rel_speed,
                -1.0, 1.0
            ))
        else:
            d2, ux2, uy2, closing2 = float(self.map_size), 0.0, 0.0, 0.0

        return (d1, ux1, uy1, vx_rel_1, vy_rel_1,
                d2, ux2, uy2, closing2)

    # ------------------------------------------------------------------ #
    # reset                                                                #
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.ship.reset(10, 10, 0)

        self.step_count    = 0
        self.prev_progress = 0

        self.waypoint_manager.reset()
        self.obstacles.reset()
        self.ocean.reset()

        x       = self.ship.x
        y       = self.ship.y
        heading = self.ship.heading

        distance_to_wp = self.waypoint_manager.distance_to_waypoint(x, y)
        _FAR = float(self.map_size)

        obs = np.array([
            x, y, heading, distance_to_wp,
            0.0, 0.0, 0.0, 0.0,
            self.ship.speed,
            _FAR, 0.0, 0.0, 0.0, 0.0,
            _FAR, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

        return obs, {}

    # ------------------------------------------------------------------ #
    # step                                                                 #
    # ------------------------------------------------------------------ #
    def step(self, action):

        self.step_count += 1

        rudder   = float(action[0])
        throttle = float(action[1])

        state = self.ship.step(rudder, throttle, self.dt)
        x, y, heading, speed, yaw_rate = state

        x, y, current = self.ocean.apply_current(x, y, self.dt)
        self.ship.x = x
        self.ship.y = y

        self.obstacles.step()

        # ── Waypoint progress ────────────────────────────────────────────
        distance_to_wp = self.waypoint_manager.distance_to_waypoint(x, y)

        reached = False
        if distance_to_wp < self.waypoint_manager.threshold:
            reached = self.waypoint_manager.check_waypoint_reached(x, y)

        current_wp = self.waypoint_manager.get_current_waypoint()

        if self.waypoint_manager.current_index == 0:
            prev_wp = self.waypoints[0]
            next_wp = self.waypoints[1]
        else:
            prev_wp = self.waypoints[self.waypoint_manager.current_index - 1]
            next_wp = current_wp

        # ── Path geometry ────────────────────────────────────────────────
        cte      = cross_track_error((x, y), prev_wp, next_wp)
        path_vec = np.array(next_wp) - np.array(prev_wp)
        path_len = np.linalg.norm(path_vec)

        if path_len < 1e-6:
            path_vec = np.array([1.0, 0.0])
        else:
            path_vec = path_vec / path_len

        ship_vec   = np.array([x, y]) - np.array(prev_wp)
        projection = float(np.clip(np.dot(ship_vec, path_vec), 0, path_len))

        lookahead_dist = float(np.clip(projection + self.lookahead, 0, path_len))
        target         = np.array(prev_wp) + lookahead_dist * path_vec

        ddx         = target[0] - x
        ddy         = target[1] - y
        angle_to_wp = np.arctan2(ddy, ddx) - heading
        angle_to_wp = float(np.arctan2(np.sin(angle_to_wp), np.cos(angle_to_wp)))

        # ── Obstacle sensing ─────────────────────────────────────────────
        ship_vx = float(speed * np.cos(heading))
        ship_vy = float(speed * np.sin(heading))

        (d1, ux1, uy1, vx_rel_1, vy_rel_1,
         d2, ux2, uy2, closing2) = self._obstacle_obs(x, y, ship_vx, ship_vy)

        # ── Observation ──────────────────────────────────────────────────
        obs = np.array([
            x, y, heading, distance_to_wp,
            angle_to_wp, cte, current[0], current[1],
            speed,
            d1, ux1, uy1, vx_rel_1, vy_rel_1,
            d2, ux2, uy2, closing2,
        ], dtype=np.float32)

        # ── Reward ───────────────────────────────────────────────────────
        #
        # DESIGN PRINCIPLE: moving toward the next waypoint must ALWAYS
        # yield a higher per-step reward than standing still, even when
        # near an obstacle.
        #
        # Progress reward: scaled up 3× so each unit of path advance
        # is worth ~0.9 reward at full speed (ship speed ≈ 3, dt = 0.1,
        # → progress ≈ 0.3 per step → reward ≈ +0.9).
        # Heading reward: kept at 0.6 max so alignment matters.
        # CTE penalty: halved — was -0.12, now -0.06.  A 5-unit CTE was
        #   costing -0.6/step which over-penalised wide paths near obstacles.
        # Rudder/yaw penalties: unchanged, small smoothness terms.
        #
        # Obstacle penalty rescaled so max per-step penalty = -1.5,
        # well below the ~+0.9 progress reward.  This means the agent
        # earns more reward by moving through the warn zone on its way
        # to the waypoint than by stopping to avoid the penalty.
        #
        # Terminal collision: -50 (down from -200).
        # -200 was so catastrophic the agent preferred to freeze and bleed
        # -13/step rather than risk a single -200 hit — it optimised for
        # "never collide" at the expense of never moving.  -50 makes
        # collision costly but not so dominant that the policy collapses.
        #
        # Waypoint bonus: 20, goal bonus: +50 on top.

        progress        = projection
        progress_reward = 3.0 * (progress - self.prev_progress)   # scaled up 3×

        if reached:
            self.prev_progress = 0
        else:
            self.prev_progress = progress

        heading_reward = 0.6 * np.cos(angle_to_wp)
        cte_penalty    = -0.06 * abs(cte)       # halved from -0.12
        rudder_penalty = -0.03 * abs(rudder)
        yaw_penalty    = -0.05 * abs(yaw_rate)

        # Graduated obstacle penalty — max -1.5/step (was -13/step)
        #   dist ≥ warn (8)    →  0
        #   warn → danger (4)  →  linear ramp  0 → -1.0
        #   dist < danger (4)  →  -1.0 + extra -0.5 spike
        obstacle_penalty = 0.0
        if d1 < self.obs_warn_radius:
            t = (self.obs_warn_radius - d1) / (
                self.obs_warn_radius - self.obs_danger_radius + 1e-6
            )
            obstacle_penalty = -1.0 * float(np.clip(t, 0.0, 1.0))

        if d1 < self.obs_danger_radius:
            obstacle_penalty -= 0.5

        waypoint_bonus = 20 if reached else 0

        reward = (
            progress_reward
            + heading_reward
            + cte_penalty
            + rudder_penalty
            + yaw_penalty
            + waypoint_bonus
            + obstacle_penalty
        )

        # ── Termination ──────────────────────────────────────────────────
        terminated = False
        info       = {}

        if d1 < self.obstacles.get_radius():
            reward    -= 50         # was -200: reduced so agent prefers
            terminated = True       # moving + occasional hit over freezing
            info["reason"] = "collision"

        elif x < 0 or x > self.map_size or y < 0 or y > self.map_size:
            reward    -= 20
            terminated = True
            info["reason"] = "boundary"

        elif self.waypoint_manager.is_navigation_complete(x, y):
            reward    += 50
            terminated = True
            info["reason"] = "goal"

        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # debug                                                                #
    # ------------------------------------------------------------------ #
    def render(self):
        x, y, heading = self.ship.get_state()
        print(f"Ship ({x:.2f},{y:.2f}) heading {heading:.2f}")