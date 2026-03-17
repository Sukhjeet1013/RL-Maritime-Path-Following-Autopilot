"""
plot_metrics.py
---------------
Reads  metrics/episode_log.json  (written by simulator.py) and generates:

  Plot 1 — Episode reward over time          (training reward curve proxy)
  Plot 2 — Goal rate vs collision rate       (rolling window)
  Plot 3 — Episode length over time          (steps per episode)
  Plot 4 — Reward component breakdown        (stacked area per episode)
  Plot 5 — Policy comparison                 (first half vs second half of log)

Output:
  metrics/plot_1_reward.png
  metrics/plot_2_goal_collision_rate.png
  metrics/plot_3_episode_length.png
  metrics/plot_4_reward_components.png
  metrics/plot_5_policy_comparison.png
  metrics/maritime_report.pdf               (all 5 combined)

Usage:
  python visualization/plot_metrics.py
  python visualization/plot_metrics.py --log metrics/episode_log.json
  python visualization/plot_metrics.py --window 20
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Style ──────────────────────────────────────────────────────────────────
DARK_BG    = "#14192B"
GRID_COLOR = "#2A3250"
TEXT_COLOR = "#C8D0E8"
TEAL       = "#1FC98E"
RED        = "#E85D5D"
AMBER      = "#F0A030"
BLUE       = "#4A9EDF"
PURPLE     = "#9B7FE8"
GRAY       = "#6B7595"

plt.rcParams.update({
    "figure.facecolor"  : DARK_BG,
    "axes.facecolor"    : DARK_BG,
    "axes.edgecolor"    : GRID_COLOR,
    "axes.labelcolor"   : TEXT_COLOR,
    "axes.titlecolor"   : TEXT_COLOR,
    "axes.grid"         : True,
    "grid.color"        : GRID_COLOR,
    "grid.linewidth"    : 0.5,
    "xtick.color"       : TEXT_COLOR,
    "ytick.color"       : TEXT_COLOR,
    "text.color"        : TEXT_COLOR,
    "legend.facecolor"  : "#1E2540",
    "legend.edgecolor"  : GRID_COLOR,
    "legend.labelcolor" : TEXT_COLOR,
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "lines.linewidth"   : 1.8,
    "savefig.facecolor" : DARK_BG,
    "savefig.dpi"       : 150,
})


# ── Helpers ────────────────────────────────────────────────────────────────
def rolling(arr, w):
    """Simple rolling mean."""
    out = []
    for i in range(len(arr)):
        start = max(0, i - w + 1)
        out.append(np.mean(arr[start:i + 1]))
    return np.array(out)


def shade(ax, x, y, color, alpha=0.15):
    """Thin line + shaded area under it."""
    ax.plot(x, y, color=color)
    ax.fill_between(x, y, alpha=alpha, color=color)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")


# ── Load data ──────────────────────────────────────────────────────────────
def load_log(path):
    if not os.path.exists(path):
        print(f"ERROR: log file not found at {path}")
        print("Run simulator.py first to generate episode data.")
        sys.exit(1)

    with open(path) as f:
        log = json.load(f)

    if len(log) < 5:
        print(f"Only {len(log)} episodes in log — run more episodes first.")
        sys.exit(1)

    episodes  = [e["episode"]        for e in log]
    outcomes  = [e["outcome"]        for e in log]
    rewards   = [e["reward"]         for e in log]
    steps     = [e["steps"]          for e in log]
    waypoints = [e.get("waypoints", 0) for e in log]
    r_prog    = [e.get("reward_progress", 0) for e in log]
    r_head    = [e.get("reward_heading",  0) for e in log]
    r_obs     = [e.get("reward_obstacle", 0) for e in log]
    r_cte     = [e.get("reward_cte",      0) for e in log]

    goals      = [1 if o == "goal"      else 0 for o in outcomes]
    collisions = [1 if o == "collision" else 0 for o in outcomes]

    return dict(
        episodes=np.array(episodes),
        outcomes=outcomes,
        rewards=np.array(rewards, dtype=float),
        steps=np.array(steps, dtype=float),
        waypoints=np.array(waypoints, dtype=float),
        r_prog=np.array(r_prog, dtype=float),
        r_head=np.array(r_head, dtype=float),
        r_obs=np.array(r_obs,  dtype=float),
        r_cte=np.array(r_cte,  dtype=float),
        goals=np.array(goals, dtype=float),
        collisions=np.array(collisions, dtype=float),
    )


# ── Plot 1 — Episode reward ────────────────────────────────────────────────
def plot_reward(d, window, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ep = d["episodes"]
    r  = d["rewards"]

    ax.scatter(ep, r, color=GRAY, alpha=0.3, s=12, zorder=2, label="Raw reward")
    shade(ax, ep, rolling(r, window), TEAL)
    ax.plot(ep, rolling(r, window), color=TEAL, label=f"Rolling mean (w={window})", zorder=3)
    ax.axhline(0, color=GRID_COLOR, linewidth=1)

    ax.set_title("Episode reward over time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_1_reward.png")
    save(fig, path)
    plt.close(fig)
    return path


# ── Plot 2 — Goal / collision rate ────────────────────────────────────────
def plot_rates(d, window, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ep = d["episodes"]

    goal_rate = rolling(d["goals"],      window) * 100
    coll_rate = rolling(d["collisions"], window) * 100

    shade(ax, ep, goal_rate, TEAL)
    ax.plot(ep, goal_rate, color=TEAL, label=f"Goal rate %   (rolling {window})")

    shade(ax, ep, coll_rate, RED)
    ax.plot(ep, coll_rate, color=RED,  label=f"Collision rate % (rolling {window})")

    # final values
    ax.axhline(goal_rate[-1], color=TEAL, linewidth=0.8,
               linestyle="--", alpha=0.6,
               label=f"Final goal rate  {goal_rate[-1]:.1f}%")
    ax.axhline(80, color=AMBER, linewidth=1.0,
               linestyle=":", alpha=0.8, label="80% target")

    ax.set_ylim(0, 105)
    ax.set_title("Goal rate vs collision rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_2_goal_collision_rate.png")
    save(fig, path)
    plt.close(fig)
    return path


# ── Plot 3 — Episode length ───────────────────────────────────────────────
def plot_steps(d, window, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ep = d["episodes"]
    st = d["steps"]

    # colour-code by outcome
    for i, (e, s, o) in enumerate(zip(ep, st, d["outcomes"])):
        c = TEAL if o == "goal" else (RED if o == "collision" else GRAY)
        ax.scatter(e, s, color=c, alpha=0.5, s=14, zorder=2)

    ax.plot(ep, rolling(st, window), color=BLUE,
            label=f"Rolling mean steps (w={window})", zorder=3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,  markersize=7, label="Goal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,   markersize=7, label="Collision"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAY,  markersize=7, label="Other"),
        Line2D([0], [0], color=BLUE, linewidth=2,                                    label=f"Rolling mean (w={window})"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_title("Episode length over time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_3_episode_length.png")
    save(fig, path)
    plt.close(fig)
    return path


# ── Plot 4 — Reward component breakdown ──────────────────────────────────
def plot_components(d, window, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    ep = d["episodes"]

    pairs = [
        (axes[0, 0], d["r_prog"], TEAL,   "Progress reward"),
        (axes[0, 1], d["r_head"], BLUE,   "Heading reward"),
        (axes[1, 0], d["r_obs"],  RED,    "Obstacle penalty"),
        (axes[1, 1], d["r_cte"], PURPLE,  "CTE penalty"),
    ]

    for ax, values, color, title in pairs:
        ax.scatter(ep, values, color=color, alpha=0.25, s=10, zorder=2)
        ax.plot(ep, rolling(values, window), color=color,
                label=f"Rolling mean (w={window})", zorder=3)
        ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Reward")
        ax.legend(fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel("Episode")

    fig.suptitle("Reward component breakdown per episode", y=1.01)
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_4_reward_components.png")
    save(fig, path)
    plt.close(fig)
    return path


# ── Plot 5 — Policy comparison: first half vs second half ─────────────────
def plot_comparison(d, out_dir):
    n    = len(d["episodes"])
    mid  = n // 2

    def stats(arr, mask=None):
        a = arr if mask is None else arr[mask]
        return float(np.mean(a)), float(np.std(a))

    first = np.arange(n) < mid
    second = ~first

    metrics = {
        "Goal rate %"       : (d["goals"]      * 100, True),
        "Collision rate %"  : (d["collisions"] * 100, False),
        "Mean reward"       : (d["rewards"],           True),
        "Mean steps"        : (d["steps"],             True),
        "Waypoints/episode" : (d["waypoints"],         True),
    }

    labels  = list(metrics.keys())
    first_m  = []
    second_m = []

    for label, (arr, _) in metrics.items():
        first_m.append(stats(arr, first)[0])
        second_m.append(stats(arr, second)[0])

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - width / 2, first_m,  width, label=f"First {mid} episodes",
                   color=GRAY, alpha=0.85)
    bars2 = ax.bar(x + width / 2, second_m, width, label=f"Last {n - mid} episodes",
                   color=TEAL, alpha=0.85)

    # value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=TEAL)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Policy comparison — first half vs second half of evaluation")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(out_dir, "plot_5_policy_comparison.png")
    save(fig, path)
    plt.close(fig)
    return path


# ── PDF report ─────────────────────────────────────────────────────────────
def build_pdf(png_paths, out_dir, d, window):
    pdf_path = os.path.join(out_dir, "maritime_report.pdf")
    n = len(d["episodes"])
    goal_rate = float(np.mean(d["goals"]))  * 100
    coll_rate = float(np.mean(d["collisions"])) * 100

    with PdfPages(pdf_path) as pdf:

        # ── Cover page ──────────────────────────────
        fig = plt.figure(figsize=(10, 7))
        fig.patch.set_facecolor(DARK_BG)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor(DARK_BG)

        ax.text(0.5, 0.78, "RL Maritime Autopilot",
                ha="center", fontsize=26, fontweight="bold", color=TEAL,
                transform=ax.transAxes)
        ax.text(0.5, 0.68, "Performance Metrics Report",
                ha="center", fontsize=18, color=TEXT_COLOR,
                transform=ax.transAxes)

        summary = [
            f"Total episodes evaluated : {n}",
            f"Goal rate                : {goal_rate:.1f}%",
            f"Collision rate           : {coll_rate:.1f}%",
            f"Mean episode reward      : {np.mean(d['rewards']):.1f}",
            f"Mean episode length      : {np.mean(d['steps']):.0f} steps",
            f"Mean waypoints / episode : {np.mean(d['waypoints']):.2f} / 3",
            f"Rolling window used      : {window} episodes",
        ]
        for i, line in enumerate(summary):
            ax.text(0.5, 0.52 - i * 0.06, line,
                    ha="center", fontsize=13, color=TEXT_COLOR,
                    transform=ax.transAxes,
                    fontfamily="monospace")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── One plot per page ────────────────────────
        for png_path in png_paths:
            if not os.path.exists(png_path):
                continue
            img = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.set_axis_off()
            fig.patch.set_facecolor(DARK_BG)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Saved → {pdf_path}")
    return pdf_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot maritime RL metrics")
    parser.add_argument("--log",    default="metrics/episode_log.json",
                        help="Path to episode_log.json")
    parser.add_argument("--window", type=int, default=15,
                        help="Rolling average window size (default 15)")
    parser.add_argument("--out",    default="metrics",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nLoading log: {args.log}")
    d = load_log(args.log)
    n = len(d["episodes"])
    print(f"  {n} episodes found")
    print(f"  Goal rate     : {np.mean(d['goals'])*100:.1f}%")
    print(f"  Collision rate: {np.mean(d['collisions'])*100:.1f}%")
    print(f"  Mean reward   : {np.mean(d['rewards']):.1f}")
    print(f"\nGenerating plots (window={args.window})...")

    png_paths = [
        plot_reward(d,     args.window, args.out),
        plot_rates(d,      args.window, args.out),
        plot_steps(d,      args.window, args.out),
        plot_components(d, args.window, args.out),
        plot_comparison(d,             args.out),
    ]

    print("\nBuilding PDF report...")
    build_pdf(png_paths, args.out, d, args.window)

    print(f"\nDone. All files saved to: {args.out}/")


if __name__ == "__main__":
    main()