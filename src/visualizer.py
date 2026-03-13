"""
visualizer.py
-------------
3D visualization for the occupancy grid, planned paths,
drone trajectory animation, and algorithm comparison plots.

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches


# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "obstacle":   "#E74C3C",
    "free":       "#ECF0F1",
    "astar":      "#2ECC71",
    "rrt_star":   "#3498DB",
    "drone":      "#F39C12",
    "start":      "#27AE60",
    "goal":       "#8E44AD",
    "dynamic_obs":"#FF6B35",
    "waypoint":   "#F1C40F",
}


class Visualizer3D:

    def __init__(self, grid):
        """
        Args:
            grid: OccupancyGrid3D instance
        """
        self.grid = grid

    # ------------------------------------------------------------------
    # Core scene rendering
    # ------------------------------------------------------------------

    def _setup_axes(self, title="3D Drone Path Planning"):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.grid.x_size)
        ax.set_ylim(0, self.grid.y_size)
        ax.set_zlim(0, self.grid.z_size)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z / Altitude (m)", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        return fig, ax

    def _draw_obstacles(self, ax, alpha=0.25):
        """Render obstacle voxels as semi-transparent cubes."""
        obs = np.argwhere(self.grid.grid == 1)
        if len(obs) == 0:
            return
        ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2],
                   c=COLORS["obstacle"], marker='s',
                   s=40, alpha=alpha, label="Obstacle", depthshade=True)

    def _draw_path(self, ax, path, color, label, linewidth=2.5, marker='o', markersize=4):
        if not path:
            return
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth,
                marker=marker, markersize=markersize, label=label, zorder=5)

    def _draw_endpoints(self, ax, start, goal):
        ax.scatter(*start, c=COLORS["start"], s=200, marker='^',
                   zorder=10, label="Start", edgecolors='black', linewidths=0.8)
        ax.scatter(*goal, c=COLORS["goal"], s=200, marker='*',
                   zorder=10, label="Goal", edgecolors='black', linewidths=0.8)

    # ------------------------------------------------------------------
    # Public: plot single path
    # ------------------------------------------------------------------

    def plot_path(self, path, start, goal,
                  algo_name="Path", color=None, save_path=None, show=True):
        """
        Render environment + a single planned path.
        """
        color = color or COLORS["astar"]
        fig, ax = self._setup_axes(f"Path Planning — {algo_name}")
        self._draw_obstacles(ax)
        self._draw_path(ax, path, color=color, label=algo_name)
        self._draw_endpoints(ax, start, goal)
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Viz] Saved → {save_path}")
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Public: compare two algorithms side by side
    # ------------------------------------------------------------------

    def plot_comparison(self, path_astar, path_rrt,
                        start, goal, metrics=None, save_path=None, show=True):
        """
        Side-by-side 3D view of A* path vs RRT* path with metrics table.
        """
        fig = plt.figure(figsize=(18, 7))
        fig.suptitle("Path Planning Algorithm Comparison: A* vs RRT*",
                     fontsize=15, fontweight='bold', y=1.01)

        # --- Left: A* ---
        ax1 = fig.add_subplot(121, projection='3d')
        self._configure_ax(ax1, "A* (Optimal Grid Search)")
        self._draw_obstacles(ax1)
        self._draw_path(ax1, path_astar, COLORS["astar"], "A* Path")
        self._draw_endpoints(ax1, start, goal)
        ax1.legend(fontsize=8)

        # --- Right: RRT* ---
        ax2 = fig.add_subplot(122, projection='3d')
        self._configure_ax(ax2, "RRT* (Probabilistic Sampling)")
        self._draw_obstacles(ax2)
        self._draw_path(ax2, path_rrt, COLORS["rrt_star"], "RRT* Path")
        self._draw_endpoints(ax2, start, goal)
        ax2.legend(fontsize=8)

        # --- Metrics annotation ---
        if metrics:
            self._add_metrics_text(fig, metrics)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Viz] Comparison saved → {save_path}")
        if show:
            plt.show()
        return fig

    def _configure_ax(self, ax, title):
        ax.set_xlim(0, self.grid.x_size)
        ax.set_ylim(0, self.grid.y_size)
        ax.set_zlim(0, self.grid.z_size)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)

    def _add_metrics_text(self, fig, metrics):
        text_lines = []
        for algo, m in metrics.items():
            text_lines.append(
                f"{algo}: length={m.get('path_length', '?'):.2f}m | "
                f"nodes={m.get('nodes_explored', '?')} | "
                f"time={m.get('time_ms', '?'):.1f}ms"
            )
        fig.text(0.5, -0.02, "\n".join(text_lines),
                 ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))

    # ------------------------------------------------------------------
    # Public: metrics bar charts
    # ------------------------------------------------------------------

    def plot_metrics(self, metrics_dict, save_path=None, show=True):
        """
        Bar chart comparison of key metrics across algorithms.
        metrics_dict: {"A*": {"path_length": X, "nodes_explored": Y, "time_ms": Z}, ...}
        """
        algos = list(metrics_dict.keys())
        keys = ["path_length", "nodes_explored", "time_ms"]
        titles = ["Path Length (voxels)", "Nodes Explored", "Planning Time (ms)"]
        colors = [COLORS["astar"], COLORS["rrt_star"], "#95A5A6", "#E67E22"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Algorithm Performance Comparison", fontsize=14, fontweight='bold')

        for i, (key, title) in enumerate(zip(keys, titles)):
            vals = [metrics_dict[a].get(key, 0) for a in algos]
            bars = axes[i].bar(algos, vals,
                               color=colors[:len(algos)],
                               edgecolor='black', linewidth=0.7, alpha=0.85)
            axes[i].set_title(title, fontsize=11)
            axes[i].set_ylabel(title, fontsize=9)
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() * 1.02,
                             f"{val:.1f}", ha='center', va='bottom', fontsize=10)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Viz] Metrics saved → {save_path}")
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Public: drone flight animation
    # ------------------------------------------------------------------

    def animate_flight(self, path, start, goal,
                       algo_name="Drone Flight", interval=120,
                       save_path=None, show=True):
        """
        Animate drone moving along a planned path step by step.
        """
        if not path or len(path) < 2:
            print("[Viz] Path too short to animate.")
            return

        fig, ax = self._setup_axes(f"{algo_name} — Flight Animation")
        self._draw_obstacles(ax, alpha=0.15)
        self._draw_endpoints(ax, start, goal)

        # Trail line
        trail_x, trail_y, trail_z = [], [], []
        trail_line, = ax.plot([], [], [], color=COLORS["astar"],
                              linewidth=2, alpha=0.7, label="Trail")
        drone_pt = ax.scatter([], [], [], c=COLORS["drone"],
                              s=200, marker='D', zorder=10,
                              label="Drone", edgecolors='black', linewidths=0.8)

        step_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)

        def init():
            trail_line.set_data([], [])
            trail_line.set_3d_properties([])
            return trail_line, drone_pt

        def update(frame):
            if frame >= len(path):
                frame = len(path) - 1
            x, y, z = path[frame]
            trail_x.append(x); trail_y.append(y); trail_z.append(z)
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)
            drone_pt._offsets3d = ([x], [y], [z])
            progress = (frame + 1) / len(path) * 100
            step_text.set_text(f"Step {frame+1}/{len(path)}  ({progress:.0f}%)")
            return trail_line, drone_pt, step_text

        ani = animation.FuncAnimation(
            fig, update, frames=len(path),
            init_func=init, interval=interval,
            blit=False, repeat=False
        )

        if save_path:
            writer = animation.PillowWriter(fps=10)
            ani.save(save_path, writer=writer)
            print(f"[Viz] Animation saved → {save_path}")
        if show:
            plt.show()
        return ani

    # ------------------------------------------------------------------
    # Public: replanning event visualization
    # ------------------------------------------------------------------

    def plot_replanning(self, original_path, new_path,
                        start, goal, obstacle_pos,
                        save_path=None, show=True):
        """
        Show original path, dynamic obstacle injection point, and replanned path.
        """
        fig, ax = self._setup_axes("Dynamic Replanning — Obstacle Avoidance")
        self._draw_obstacles(ax)

        # Original path (dashed)
        self._draw_path(ax, original_path, color="#95A5A6",
                        label="Original Path", linewidth=1.5)

        # New replanned path
        self._draw_path(ax, new_path, color=COLORS["rrt_star"],
                        label="Replanned Path", linewidth=2.5)

        # Dynamic obstacle highlight
        ox, oy, oz = obstacle_pos
        ax.scatter(ox, oy, oz, c=COLORS["dynamic_obs"], s=300,
                   marker='X', zorder=15, label="New Obstacle", edgecolors='black')

        self._draw_endpoints(ax, start, goal)
        ax.legend(loc='upper left', fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Viz] Replanning plot saved → {save_path}")
        if show:
            plt.show()
        return fig, ax
