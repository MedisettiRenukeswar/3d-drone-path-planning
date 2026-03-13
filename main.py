"""
main.py
-------
Entry point for the 3D Drone Path Planning project.

Usage:
    python main.py                        # run all phases with defaults
    python main.py --algo astar           # A* only
    python main.py --algo rrt             # RRT* only
    python main.py --algo compare         # side-by-side comparison
    python main.py --map random --seed 7  # different random environment
    python main.py --no-anim              # skip animation (faster)

Author: Your Name
"""

import argparse
import os
import sys
import numpy as np

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from src.environment    import OccupancyGrid3D
from src.visualizer     import Visualizer3D
from src.astar          import AStarPlanner, smooth_path
from src.rrt_star       import RRTStarPlanner
from src.replanner      import DynamicReplanner
from src.pid_controller import FlightSimulator


# ── Configuration ──────────────────────────────────────────────────────────────

GRID_SIZE   = (25, 25, 12)
START       = (1,  1,  1)
GOAL        = (23, 23, 10)
RESULTS_DIR = "results"


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_env(seed=42):
    grid = OccupancyGrid3D(*GRID_SIZE, resolution=1.0)
    grid.add_random_obstacles(num_boxes=10, num_cylinders=5, seed=seed)
    # Clear start and goal cells
    grid.grid[START[0], START[1], START[2]] = 0
    grid.grid[GOAL[0],  GOAL[1],  GOAL[2]]  = 0
    return grid

def ensure_results():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def print_result(res):
    tag = res["algorithm"]
    if res["found"]:
        print(f"\n  [{tag}] ✓ Path found")
        print(f"         Waypoints    : {len(res['path'])}")
        print(f"         Path length  : {res['path_length']:.2f} m")
        print(f"         Nodes explored: {res['nodes_explored']}")
        print(f"         Planning time : {res['time_ms']:.2f} ms")
    else:
        print(f"\n  [{tag}] ✗ Path NOT found")


# ── Phase runners ──────────────────────────────────────────────────────────────

def run_phase1(grid, viz, args):
    """Phase 1 — Environment + occupancy grid visualization."""
    print("\n" + "="*55)
    print("  PHASE 1: 3D Occupancy Grid")
    print("="*55)
    print(f"  Grid  : {grid}")
    print(f"  Start : {START}")
    print(f"  Goal  : {GOAL}")
    print(f"  Obstacle density: {grid.obstacle_density():.2%}")

    grid.save(os.path.join(RESULTS_DIR, "grid.npy"))
    grid.save_config(os.path.join(RESULTS_DIR, "grid_config.json"))

    viz.plot_path([], START, GOAL,
                  algo_name="Environment (no path)",
                  color="#888888",
                  save_path=os.path.join(RESULTS_DIR, "phase1_environment.png"),
                  show=not args.no_show)
    print("  Phase 1 complete ✓")


def run_phase2(grid, viz, args):
    """Phase 2 — A* and RRT* planning + comparison."""
    print("\n" + "="*55)
    print("  PHASE 2: Path Planning (A* vs RRT*)")
    print("="*55)

    # ── A* ──
    astar = AStarPlanner(grid, heuristic='euclidean')
    result_a = astar.plan(START, GOAL)
    print_result(result_a)

    smoothed_a = smooth_path(result_a["path"], grid) if result_a["found"] else []
    if smoothed_a:
        print(f"         Smoothed to  : {len(smoothed_a)} waypoints")

    # ── RRT* ──
    rrt = RRTStarPlanner(grid, max_iter=3000, step_size=1.5, seed=0)
    result_r = rrt.plan(START, GOAL)
    print_result(result_r)

    # ── Individual plots ──
    if result_a["found"]:
        from src.visualizer import COLORS
        viz.plot_path(result_a["path"], START, GOAL,
                      algo_name="A*",
                      color=COLORS["astar"],
                      save_path=os.path.join(RESULTS_DIR, "phase2_astar.png"),
                      show=not args.no_show)

    if result_r["found"]:
        from src.visualizer import COLORS
        viz.plot_path(result_r["path"], START, GOAL,
                      algo_name="RRT*",
                      color=COLORS["rrt_star"],
                      save_path=os.path.join(RESULTS_DIR, "phase2_rrt_star.png"),
                      show=not args.no_show)

    # ── Comparison ──
    if result_a["found"] and result_r["found"]:
        metrics = {
            "A*":   result_a,
            "RRT*": result_r,
        }
        viz.plot_comparison(
            result_a["path"], result_r["path"],
            START, GOAL, metrics=metrics,
            save_path=os.path.join(RESULTS_DIR, "phase2_comparison.png"),
            show=not args.no_show
        )
        viz.plot_metrics(
            metrics,
            save_path=os.path.join(RESULTS_DIR, "phase2_metrics.png"),
            show=not args.no_show
        )

    print("  Phase 2 complete ✓")
    return result_a, result_r, astar, rrt


def run_phase3(grid, viz, result_a, result_r, astar, rrt, args):
    """Phase 3 — Dynamic obstacle injection and replanning."""
    print("\n" + "="*55)
    print("  PHASE 3: Dynamic Replanning")
    print("="*55)

    replanner = DynamicReplanner(grid, astar, verbose=True)
    sim_result = replanner.simulate(
        START, GOAL,
        obstacle_inject_step=None,  # auto: 40% through path
        obstacle_pos=None,          # auto: on-path placement
        obstacle_size=2,
    )

    if sim_result["replanned"]:
        viz.plot_replanning(
            original_path=sim_result["initial_plan"]["path"],
            new_path=sim_result["replan"]["path"],
            start=START, goal=GOAL,
            obstacle_pos=sim_result["obstacle_pos"],
            save_path=os.path.join(RESULTS_DIR, "phase3_replanning.png"),
            show=not args.no_show
        )
        print(f"\n  Replanning summary:")
        print(f"    Initial plan  : {sim_result['initial_plan']['path_length']:.2f}m")
        print(f"    Replanned     : {sim_result['replan']['path_length']:.2f}m")
        print(f"    Replan time   : {sim_result['replan']['time_ms']:.1f}ms")

    print("  Phase 3 complete ✓")
    return sim_result


def run_phase4(grid, viz, result_a, result_r, args):
    """Phase 4 — PID waypoint tracking simulation."""
    print("\n" + "="*55)
    print("  PHASE 4: PID Flight Simulation")
    print("="*55)

    simulator = FlightSimulator(grid, dt=0.05)

    if result_a["found"]:
        print("\n  [A*] Starting flight simulation...")
        flight = simulator.fly(result_a["path"], verbose=True)
        print(f"\n  [A*] Flight result:")
        print(f"       Success      : {flight['success']}")
        print(f"       Final error  : {flight['final_error_m']} m")
        print(f"       Flight time  : {flight['total_time_s']} s")
        print(f"       Waypoints hit: {flight['waypoints_hit']}/{flight['waypoints_total']}")

        if not args.no_anim:
            viz.animate_flight(
                result_a["path"], START, GOAL,
                algo_name="A* Flight",
                save_path=os.path.join(RESULTS_DIR, "phase4_flight_astar.gif"),
                show=not args.no_show
            )

    if result_r["found"] and not args.no_anim:
        viz.animate_flight(
            result_r["path"], START, GOAL,
            algo_name="RRT* Flight",
            save_path=os.path.join(RESULTS_DIR, "phase4_flight_rrt.gif"),
            show=not args.no_show
        )

    print("  Phase 4 complete ✓")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D Drone Path Planning: A* vs RRT* with Dynamic Replanning")
    parser.add_argument("--algo",    default="compare",
                        choices=["astar", "rrt", "compare"],
                        help="Which algorithm(s) to run")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for obstacle generation")
    parser.add_argument("--no-anim", action="store_true",
                        help="Skip flight animation (faster)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open plot windows (save only)")
    args = parser.parse_args()

    ensure_results()

    print("\n" + "█"*55)
    print("  3D DRONE PATH PLANNING")
    print("  A* | RRT* | Dynamic Replanning | PID Control")
    print("█"*55)

    # Build environment
    grid = make_env(seed=args.seed)
    viz  = Visualizer3D(grid)

    # Run all phases
    run_phase1(grid, viz, args)
    result_a, result_r, astar, rrt = run_phase2(grid, viz, args)
    sim = run_phase3(grid, viz, result_a, result_r, astar, rrt, args)
    run_phase4(grid, viz, result_a, result_r, args)

    print("\n" + "█"*55)
    print("  All phases complete. Results saved to ./results/")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()
