"""
astar.py
--------
A* Search Algorithm for 3D Occupancy Grid path planning.

Key concepts:
  - f(n) = g(n) + h(n)
  - g(n): actual cost from start to node n
  - h(n): heuristic estimate from n to goal (Euclidean distance)
  - Uses a min-heap (priority queue) for efficiency

Time complexity:  O(b^d) worst case, near-optimal with good heuristic
Space complexity: O(b^d)

Author: Your Name
"""

import heapq
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Heuristics ─────────────────────────────────────────────────────────────────

def euclidean_heuristic(a, b):
    """Straight-line distance in 3D — admissible and consistent."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def manhattan_heuristic(a, b):
    """Sum of absolute differences — faster but less accurate in 3D."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def diagonal_heuristic(a, b):
    """Chebyshev distance — suitable for 26-connected grids."""
    dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1]); dz = abs(a[2]-b[2])
    return max(dx, dy, dz)


# ── A* Node ────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class Node:
    f: float                        # priority in heap
    g: float = field(compare=False) # cost from start
    pos: tuple = field(compare=False)
    parent: Optional[object] = field(default=None, compare=False)


# ── A* Planner ─────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    A* path planner operating on an OccupancyGrid3D.

    Usage:
        planner = AStarPlanner(grid, heuristic='euclidean')
        result  = planner.plan(start=(0,0,0), goal=(18,18,8))
    """

    HEURISTICS = {
        'euclidean': euclidean_heuristic,
        'manhattan': manhattan_heuristic,
        'diagonal':  diagonal_heuristic,
    }

    def __init__(self, grid, heuristic='euclidean', weight=1.0):
        """
        Args:
            grid:      OccupancyGrid3D
            heuristic: 'euclidean' | 'manhattan' | 'diagonal'
            weight:    inflation factor for heuristic (w=1 → standard A*,
                       w>1 → weighted A* — faster but suboptimal)
        """
        self.grid = grid
        self.h = self.HEURISTICS.get(heuristic, euclidean_heuristic)
        self.weight = weight

    def plan(self, start, goal):
        """
        Run A* from start to goal.

        Returns:
            dict with keys:
              path           - list of (x,y,z) tuples, or []
              found          - bool
              path_length    - Euclidean length of path
              nodes_explored - number of nodes popped from open list
              time_ms        - planning time in milliseconds
        """
        t0 = time.perf_counter()

        start = tuple(start); goal = tuple(goal)

        # Sanity checks
        if not self.grid.is_free(*start):
            return self._failure("Start is inside an obstacle.", t0)
        if not self.grid.is_free(*goal):
            return self._failure("Goal is inside an obstacle.", t0)

        open_list  = []   # min-heap
        closed_set = set()
        g_score    = {start: 0.0}
        came_from  = {}

        h0 = self.weight * self.h(start, goal)
        start_node = Node(f=h0, g=0.0, pos=start)
        heapq.heappush(open_list, start_node)

        nodes_explored = 0

        while open_list:
            current = heapq.heappop(open_list)

            if current.pos in closed_set:
                continue
            closed_set.add(current.pos)
            nodes_explored += 1

            # ── Goal reached ──
            if current.pos == goal:
                path = self._reconstruct_path(came_from, goal)
                elapsed = (time.perf_counter() - t0) * 1000
                return {
                    "path":            path,
                    "found":           True,
                    "path_length":     self._path_length(path),
                    "nodes_explored":  nodes_explored,
                    "time_ms":         elapsed,
                    "algorithm":       "A*",
                }

            # ── Expand neighbors ──
            for (neighbor, step_cost) in self.grid.get_neighbors(*current.pos):
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current.pos] + step_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current.pos
                    h = self.weight * self.h(neighbor, goal)
                    f = tentative_g + h
                    heapq.heappush(open_list, Node(f=f, g=tentative_g, pos=neighbor))

        elapsed = (time.perf_counter() - t0) * 1000
        return self._failure("No path found — goal unreachable.", t0,
                             nodes_explored=nodes_explored, elapsed=elapsed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _path_length(self, path):
        if len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(path)):
            a, b = np.array(path[i-1]), np.array(path[i])
            length += np.linalg.norm(b - a)
        return round(length * self.grid.resolution, 4)

    def _failure(self, msg, t0, nodes_explored=0, elapsed=None):
        if elapsed is None:
            elapsed = (time.perf_counter() - t0) * 1000
        print(f"[A*] {msg}")
        return {
            "path":           [],
            "found":          False,
            "path_length":    0,
            "nodes_explored": nodes_explored,
            "time_ms":        elapsed,
            "algorithm":      "A*",
        }


# ── Path post-processing ───────────────────────────────────────────────────────

def smooth_path(path, grid, max_iterations=100):
    """
    Path shortcutting: try to connect non-adjacent waypoints directly
    if the straight segment is collision-free.

    Returns a shortened path list.
    """
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to jump as far forward as possible
        j = len(path) - 1
        while j > i + 1:
            if _line_of_sight(path[i], path[j], grid):
                break
            j -= 1
        smoothed.append(path[j])
        i = j

    return smoothed


def _line_of_sight(p1, p2, grid, samples=20):
    """
    Bresenham-style line check between two 3D points.
    Returns True if all interpolated voxels are free.
    """
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    for t in np.linspace(0, 1, samples):
        pt = p1 + t * (p2 - p1)
        xi, yi, zi = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
        if not grid.is_free(xi, yi, zi):
            return False
    return True
