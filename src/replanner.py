"""
replanner.py
------------
Dynamic obstacle injection and real-time path replanning.

Simulates a mid-flight scenario where a new obstacle appears on
the drone's current path and the system must replan using either
A* or RRT*.

Author: Your Name
"""

import numpy as np
import time
from copy import deepcopy


class DynamicReplanner:
    """
    Wraps a path planner with dynamic obstacle handling.

    Workflow:
      1. Plan initial path (start → goal)
      2. Drone begins traversal
      3. At a given step, inject a new obstacle
      4. Detect that current path is blocked
      5. Replan from current drone position to goal
      6. Resume flight on new path
    """

    def __init__(self, grid, planner, verbose=True):
        """
        Args:
            grid:    OccupancyGrid3D (will be modified in-place)
            planner: AStarPlanner or RRTStarPlanner
            verbose: print event log
        """
        self.grid    = grid
        self.planner = planner
        self.verbose = verbose
        self.events  = []  # log of planning events

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def simulate(self, start, goal,
                 obstacle_inject_step=None,
                 obstacle_pos=None,
                 obstacle_size=2):
        """
        Full simulation: initial plan → fly → obstacle → replan → fly.

        Args:
            start:                 (x,y,z) start position
            goal:                  (x,y,z) goal position
            obstacle_inject_step:  step index at which to inject obstacle
                                   (default: 40% through path)
            obstacle_pos:          (x,y,z) where to place new obstacle
                                   (default: auto-placed on path)
            obstacle_size:         size of box obstacle in voxels

        Returns:
            SimulationResult dict
        """
        result = {
            "start":          start,
            "goal":           goal,
            "initial_plan":   None,
            "replan":         None,
            "full_trajectory":None,
            "obstacle_pos":   None,
            "replanned":      False,
            "events":         [],
        }

        # ── Step 1: Initial plan ──────────────────────────────────────
        self._log("Planning initial path...")
        initial = self.planner.plan(start, goal)

        if not initial["found"]:
            self._log("ERROR: Initial path not found.")
            result["events"] = self.events
            return result

        result["initial_plan"] = initial
        path = initial["path"]
        self._log(f"Initial path found: {len(path)} waypoints, "
                  f"{initial['path_length']:.2f}m, "
                  f"{initial['time_ms']:.1f}ms")

        # ── Step 2: Determine injection point ────────────────────────
        if obstacle_inject_step is None:
            obstacle_inject_step = max(1, int(len(path) * 0.4))
        obstacle_inject_step = min(obstacle_inject_step, len(path) - 2)

        # ── Step 3: Choose obstacle position ─────────────────────────
        if obstacle_pos is None:
            # Place obstacle a few steps ahead of injection point
            look_ahead = min(obstacle_inject_step + 3, len(path) - 1)
            obstacle_pos = path[look_ahead]

        ox, oy, oz = obstacle_pos
        result["obstacle_pos"] = obstacle_pos

        # ── Step 4: Simulate flight until injection step ──────────────
        flown_segment = path[:obstacle_inject_step + 1]
        drone_pos     = path[obstacle_inject_step]
        self._log(f"Drone at step {obstacle_inject_step}: position {drone_pos}")

        # ── Step 5: Inject obstacle ───────────────────────────────────
        self.grid.inject_dynamic_obstacle(ox, oy, oz, size=obstacle_size)
        self._log(f"⚠  New obstacle injected at {obstacle_pos} (size={obstacle_size})")

        # ── Step 6: Check if path is still clear ─────────────────────
        remaining_path = path[obstacle_inject_step:]
        if self.grid.path_is_clear(remaining_path):
            self._log("Path still clear — no replanning needed.")
            result["full_trajectory"] = path
            result["events"] = self.events
            return result

        self._log("Path BLOCKED — triggering replan...")

        # ── Step 7: Replan from current drone position ────────────────
        replan_result = self.planner.plan(drone_pos, goal)
        result["replan"] = replan_result

        if not replan_result["found"]:
            self._log("ERROR: Replanning failed — no alternative path found.")
        else:
            result["replanned"]       = True
            result["full_trajectory"] = flown_segment + replan_result["path"][1:]
            self._log(f"Replanned path: {len(replan_result['path'])} waypoints, "
                      f"{replan_result['path_length']:.2f}m, "
                      f"{replan_result['time_ms']:.1f}ms")

        result["events"] = self.events
        return result

    # ------------------------------------------------------------------
    # Comparison across algorithms
    # ------------------------------------------------------------------

    def benchmark_replanning(self, start, goal,
                              planners_dict,
                              obstacle_inject_step=None,
                              obstacle_pos=None):
        """
        Run the same dynamic scenario with multiple planners and compare.

        Args:
            planners_dict: {"A*": AStarPlanner, "RRT*": RRTStarPlanner}

        Returns:
            dict of algorithm name → SimulationResult
        """
        results = {}
        for name, planner in planners_dict.items():
            self._log(f"\n{'='*40}")
            self._log(f"  Benchmarking: {name}")
            self._log(f"{'='*40}")

            # Each run needs a fresh grid (deep copy to avoid contamination)
            from src.environment import OccupancyGrid3D
            fresh_grid = OccupancyGrid3D(
                self.grid.x_size,
                self.grid.y_size,
                self.grid.z_size,
                self.grid.resolution
            )
            fresh_grid.grid = self.grid.grid.copy()

            planner.grid = fresh_grid
            replanner    = DynamicReplanner(fresh_grid, planner, verbose=self.verbose)

            res = replanner.simulate(
                start, goal,
                obstacle_inject_step=obstacle_inject_step,
                obstacle_pos=obstacle_pos,
            )
            res["algorithm"] = name
            results[name]    = res

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.events.append(entry)
        if self.verbose:
            print(entry)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @staticmethod
    def compare_summary(benchmark_results):
        """
        Print a formatted summary table from benchmark_replanning results.
        """
        print("\n" + "="*60)
        print(f"{'DYNAMIC REPLANNING BENCHMARK':^60}")
        print("="*60)
        header = f"{'Algorithm':<12} {'Init(ms)':>9} {'Replan(ms)':>11} {'Init Len':>10} {'New Len':>9} {'Replanned':>10}"
        print(header)
        print("-"*60)

        for name, res in benchmark_results.items():
            init_t   = res["initial_plan"]["time_ms"]  if res["initial_plan"]  else 0
            replan_t = res["replan"]["time_ms"]        if res["replan"]        else 0
            init_len = res["initial_plan"]["path_length"] if res["initial_plan"] else 0
            new_len  = res["replan"]["path_length"]    if res["replan"]        else 0
            did_rep  = "Yes" if res["replanned"] else "No"
            print(f"{name:<12} {init_t:>9.1f} {replan_t:>11.1f} "
                  f"{init_len:>10.2f} {new_len:>9.2f} {did_rep:>10}")

        print("="*60)
