import sys
import os
import json
import time
import pygame
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.maritime_env import MaritimePathEnv


pygame.init()

WIDTH  = 900
HEIGHT = 700
SCALE  = 6
FPS    = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Maritime Autopilot Simulator")

clock = pygame.time.Clock()
font  = pygame.font.SysFont("Arial", 16)


# ------------------------------------------------
# Load env + model
# ------------------------------------------------
env = DummyVecEnv([lambda: MaritimePathEnv()])
env = VecNormalize.load("models/env_normalization.pkl", env)
env.training    = False
env.norm_reward = False

base_env = env.venv.envs[0]

model = PPO.load("models/best/best_model.zip")

obs = env.reset()

# ------------------------------------------------
# Metrics log
# ------------------------------------------------
os.makedirs("metrics", exist_ok=True)
LOG_PATH = "metrics/episode_log.json"

# Load existing log so multiple runs accumulate
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        episode_log = json.load(f)
else:
    episode_log = []

# ── Per-episode accumulators ──────────────────────
ep_reward          = 0.0
ep_steps           = 0
ep_waypoints       = 0

# reward component accumulators
ep_progress_sum    = 0.0
ep_heading_sum     = 0.0
ep_obstacle_sum    = 0.0
ep_cte_sum         = 0.0

# running totals for HUD
total_goals      = 0
total_collisions = 0
total_episodes   = 0

# trajectory for drawing
trajectory = []


# ------------------------------------------------
# Coordinate helpers
# ------------------------------------------------
def world_to_screen(x, y):
    return int(x * SCALE), int(HEIGHT - y * SCALE)


def draw_ship(x, y, heading):
    sx, sy = world_to_screen(x, y)
    size   = 10
    p1 = (int(sx + size * np.cos(heading)),       int(sy - size * np.sin(heading)))
    p2 = (int(sx + size * np.cos(heading + 2.5)), int(sy - size * np.sin(heading + 2.5)))
    p3 = (int(sx + size * np.cos(heading - 2.5)), int(sy - size * np.sin(heading - 2.5)))
    pygame.draw.polygon(screen, (50, 200, 255), [p1, p2, p3])
    hx = int(sx + 20 * np.cos(heading))
    hy = int(sy - 20 * np.sin(heading))
    pygame.draw.line(screen, (255, 255, 255), (sx, sy), (hx, hy), 2)


# ------------------------------------------------
# Helper: compute reward components for logging
# These mirror the env reward function exactly so
# we can decompose each episode's reward.
# ------------------------------------------------
def decompose_reward(base_env, angle_to_wp, cte, d1, rudder, yaw_rate):
    heading_r  = 0.6 * np.cos(angle_to_wp)
    cte_p      = -0.06 * abs(cte)
    obs_p      = 0.0
    warn       = base_env.obs_warn_radius
    danger     = base_env.obs_danger_radius
    if d1 < warn:
        t     = (warn - d1) / (warn - danger + 1e-6)
        obs_p = -1.0 * float(np.clip(t, 0.0, 1.0))
    if d1 < danger:
        obs_p -= 0.5
    return heading_r, cte_p, obs_p


# ------------------------------------------------
# Simulation loop
# ------------------------------------------------
running = True
prev_wp_index = 0

while running:

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # ── Read true world state ─────────────────────
    ship_x  = float(base_env.ship.x)
    ship_y  = float(base_env.ship.y)
    heading = float(base_env.ship.heading)
    speed   = float(base_env.ship.speed)
    d1      = float(base_env.obstacles.get_all_obstacle_info()[0][0] if
                    hasattr(base_env.obstacles, 'get_all_obstacle_info') else
                    base_env.map_size)

    # ── Accumulate episode metrics ────────────────
    ep_reward += float(reward[0])
    ep_steps  += 1

    # waypoints reached this episode
    cur_wp = base_env.waypoint_manager.current_index
    if cur_wp > prev_wp_index:
        ep_waypoints  += (cur_wp - prev_wp_index)
        prev_wp_index  = cur_wp

    # reward component approximation
    angle_to_wp = 0.0   # approximate — exact value inside env not exposed here
    cte         = 0.0
    rudder      = float(action[0][0])
    yaw_rate    = float(base_env.ship.yaw_rate)

    h_r, cte_p, obs_p = decompose_reward(base_env, angle_to_wp, cte, d1,
                                          rudder, yaw_rate)
    ep_heading_sum  += h_r
    ep_cte_sum      += cte_p
    ep_obstacle_sum += obs_p
    ep_progress_sum += float(reward[0]) - h_r - cte_p - obs_p  # remainder = progress

    trajectory.append((ship_x, ship_y))
    if len(trajectory) > 2000:
        trajectory.pop(0)

    # ── Draw ──────────────────────────────────────
    screen.fill((20, 30, 60))

    waypoints = base_env.waypoints
    for i in range(len(waypoints) - 1):
        x1, y1 = world_to_screen(*waypoints[i])
        x2, y2 = world_to_screen(*waypoints[i + 1])
        pygame.draw.line(screen, (180, 180, 180), (x1, y1), (x2, y2), 2)

    for i, wp in enumerate(waypoints):
        wx, wy = world_to_screen(*wp)
        color  = (255, 255, 0) if i == base_env.waypoint_manager.current_index else (255, 80, 80)
        pygame.draw.circle(screen, color, (wx, wy), 7)

    if len(trajectory) > 2:
        for i in range(len(trajectory) - 1):
            x1, y1 = world_to_screen(*trajectory[i])
            x2, y2 = world_to_screen(*trajectory[i + 1])
            pygame.draw.line(screen, (80, 200, 255), (x1, y1), (x2, y2), 2)

    obs_radius      = int(base_env.obstacles.get_radius() * SCALE)
    warn_radius_px  = int(base_env.obs_warn_radius   * SCALE)
    danger_radius_px = int(base_env.obs_danger_radius * SCALE)

    for ox, oy in base_env.obstacles.get_positions():
        sx, sy = world_to_screen(ox, oy)
        pygame.draw.circle(screen, (200, 200,  60), (sx, sy), warn_radius_px,   1)
        pygame.draw.circle(screen, (255, 140,   0), (sx, sy), danger_radius_px, 1)
        pygame.draw.circle(screen, (255, 150,   0), (sx, sy), obs_radius)
        pygame.draw.circle(screen, (255,  80,   0), (sx, sy), obs_radius, 1)

    draw_ship(ship_x, ship_y, heading)

    lines = [
        f"Waypoint : {base_env.waypoint_manager.current_index + 1} / {len(waypoints)}",
        f"Position : x={ship_x:.1f}  y={ship_y:.1f}",
        f"Heading  : {np.degrees(heading):.1f}°",
        f"Goals    : {total_goals}   Collisions: {total_collisions}   Episodes: {total_episodes}",
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, (255, 255, 255)), (10, 10 + i * 20))

    pygame.display.flip()

    # ── Episode end ───────────────────────────────
    if done[0]:

        reason = info[0].get("reason", "unknown")
        total_episodes += 1

        if reason == "goal":
            total_goals += 1
            print(f"[{total_episodes:>3}] Goal reached!          reward={ep_reward:>8.1f}  steps={ep_steps}")
        elif reason == "collision":
            total_collisions += 1
            print(f"[{total_episodes:>3}] Collision!             reward={ep_reward:>8.1f}  steps={ep_steps}")
        elif reason == "boundary":
            print(f"[{total_episodes:>3}] Out of bounds!         reward={ep_reward:>8.1f}  steps={ep_steps}")
        else:
            print(f"[{total_episodes:>3}] Truncated              reward={ep_reward:>8.1f}  steps={ep_steps}")

        # ── Save episode record ───────────────────
        episode_log.append({
            "episode"       : total_episodes,
            "outcome"       : reason,
            "reward"        : round(ep_reward, 2),
            "steps"         : ep_steps,
            "waypoints"     : ep_waypoints,
            "reward_progress" : round(ep_progress_sum, 2),
            "reward_heading"  : round(ep_heading_sum,  2),
            "reward_obstacle" : round(ep_obstacle_sum, 2),
            "reward_cte"      : round(ep_cte_sum,      2),
            "timestamp"     : round(time.time(), 1),
        })

        with open(LOG_PATH, "w") as f:
            json.dump(episode_log, f, indent=2)

        # reset accumulators
        ep_reward       = 0.0
        ep_steps        = 0
        ep_waypoints    = 0
        ep_progress_sum = 0.0
        ep_heading_sum  = 0.0
        ep_obstacle_sum = 0.0
        ep_cte_sum      = 0.0
        prev_wp_index   = 0

        pygame.time.wait(800)
        trajectory.clear()
        obs = env.reset()

pygame.quit()
print(f"\nSession complete — {total_episodes} episodes logged to {LOG_PATH}")
print(f"Goal rate: {total_goals}/{total_episodes} = {100*total_goals/max(total_episodes,1):.1f}%")
print("Run plot_metrics.py to generate charts.")