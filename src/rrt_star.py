"""
rrt_star.py
-----------
RRT* (Rapidly-exploring Random Trees — Optimal) for 3D path planning.

Key concepts:
  - Samples random configurations in C-space
  - Extends tree toward samples
  - Rewires nearby nodes to maintain cost-optimality
  - Asymptotically optimal (unlike basic RRT)

Reference: Karaman & Frazzoli (2011), "Sampling-based Algorithms for
           Optimal Motion Planning"

Author: Your Name
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, List


# ── Tree Node ──────────────────────────────────────────────────────────────────

@dataclass
class RRTNode:
    pos: tuple
    parent: Optional['RRTNode'] = field(default=None, repr=False)
    cost: float = 0.0          # cumulative cost from root
    children: List = field(default_factory=list, repr=False)

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return self.pos == other.pos


# ── RRT* Planner ───────────────────────────────────────────────────────────────

class RRTStarPlanner:
    """
    RRT* path planner operating on an OccupancyGrid3D.

    Key difference from RRT:
      After adding a new node, we check all nearby nodes and rewire
      the tree if going through the new node gives a lower cost.
      This guarantees asymptotic optimality.

    Usage:
        planner = RRTStarPlanner(grid, max_iter=2000, step_size=1.5)
        result  = planner.plan(start=(0,0,0), goal=(18,18,8))
    """

    def __init__(self, grid,
                 max_iter=2000,
                 step_size=1.5,
                 goal_sample_rate=0.15,
                 rewire_radius=None,
                 seed=None):
        """
        Args:
            grid:             OccupancyGrid3D
            max_iter:         maximum sampling iterations
            step_size:        max extension distance per step (voxels)
            goal_sample_rate: probability of sampling the goal directly
            rewire_radius:    neighborhood radius for rewiring
                              (default: auto-computed from grid size)
            seed:             random seed for reproducibility
        """
        self.grid = grid
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.rng = np.random.default_rng(seed)

        # RRT* near-neighbor radius (gamma * log(n)/n)^(1/d)
        # Simplified: use a fixed radius scaled to grid size
        d = np.sqrt(grid.x_size**2 + grid.y_size**2 + grid.z_size**2)
        self.rewire_radius = rewire_radius or (step_size * 3.0)

        self.nodes: List[RRTNode] = []

    def plan(self, start, goal):
        """
        Run RRT* from start toward goal.

        Returns:
            dict with path, found, path_length, nodes_explored, time_ms
        """
        t0 = time.perf_counter()
        start = tuple(int(v) for v in start)
        goal  = tuple(int(v) for v in goal)

        if not self.grid.is_free(*start):
            return self._failure("Start is inside an obstacle.", t0)
        if not self.grid.is_free(*goal):
            return self._failure("Goal is inside an obstacle.", t0)

        # Initialise tree
        self.nodes = [RRTNode(pos=start, cost=0.0)]
        goal_node = None
        best_goal_cost = float('inf')

        for iteration in range(self.max_iter):
            # 1. Sample random point
            q_rand = self._sample(goal)

            # 2. Find nearest node in tree
            q_near = self._nearest(q_rand)

            # 3. Steer toward q_rand (bounded by step_size)
            q_new_pos = self._steer(q_near.pos, q_rand)

            # 4. Collision check
            if not self._is_collision_free(q_near.pos, q_new_pos):
                continue

            # 5. Find neighbors within rewire radius
            neighbors = self._near_nodes(q_new_pos)

            # 6. Choose best parent (lowest cost to q_new)
            q_new = self._choose_parent(q_new_pos, neighbors, q_near)

            # 7. Add node to tree
            self.nodes.append(q_new)
            if q_new.parent:
                q_new.parent.children.append(q_new)

            # 8. Rewire tree (RRT* key step)
            self._rewire(q_new, neighbors)

            # 9. Check if we reached the goal
            if self._dist(q_new.pos, goal) <= self.step_size:
                if self._is_collision_free(q_new.pos, goal):
                    goal_cost = q_new.cost + self._dist(q_new.pos, goal)
                    if goal_cost < best_goal_cost:
                        best_goal_cost = goal_cost
                        goal_node = RRTNode(pos=goal,
                                            parent=q_new,
                                            cost=goal_cost)

        elapsed = (time.perf_counter() - t0) * 1000

        if goal_node is None:
            return self._failure("Goal not reached within max_iter.", t0,
                                 len(self.nodes), elapsed)

        path = self._extract_path(goal_node)
        return {
            "path":           path,
            "found":          True,
            "path_length":    self._path_length(path),
            "nodes_explored": len(self.nodes),
            "time_ms":        elapsed,
            "algorithm":      "RRT*",
        }

    # ------------------------------------------------------------------
    # RRT* core operations
    # ------------------------------------------------------------------

    def _sample(self, goal):
        """Biased sampling: goal_sample_rate chance of picking the goal."""
        if self.rng.random() < self.goal_sample_rate:
            return goal
        return (
            int(self.rng.integers(0, self.grid.x_size)),
            int(self.rng.integers(0, self.grid.y_size)),
            int(self.rng.integers(0, self.grid.z_size)),
        )

    def _nearest(self, q):
        """Return tree node closest to q."""
        return min(self.nodes, key=lambda n: self._dist(n.pos, q))

    def _steer(self, from_pos, to_pos):
        """Move from from_pos toward to_pos by at most step_size."""
        diff = np.array(to_pos) - np.array(from_pos, dtype=float)
        dist = np.linalg.norm(diff)
        if dist <= self.step_size:
            new_pos = to_pos
        else:
            direction = diff / dist
            new_pos = np.array(from_pos) + direction * self.step_size

        # Clamp to grid bounds and convert to int
        x = int(np.clip(round(new_pos[0]), 0, self.grid.x_size - 1))
        y = int(np.clip(round(new_pos[1]), 0, self.grid.y_size - 1))
        z = int(np.clip(round(new_pos[2]), 0, self.grid.z_size - 1))
        return (x, y, z)

    def _is_collision_free(self, p1, p2, samples=15):
        """Linear interpolation collision check between two points."""
        p1a = np.array(p1, dtype=float)
        p2a = np.array(p2, dtype=float)
        for t in np.linspace(0, 1, samples):
            pt = p1a + t * (p2a - p1a)
            xi, yi, zi = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
            if not self.grid.is_free(xi, yi, zi):
                return False
        return True

    def _near_nodes(self, pos):
        """Return all tree nodes within rewire_radius of pos."""
        return [n for n in self.nodes
                if self._dist(n.pos, pos) <= self.rewire_radius]

    def _choose_parent(self, q_new_pos, neighbors, q_near):
        """
        Select the parent that gives the minimum cost to q_new.
        This is what makes RRT* find better paths than RRT.
        """
        best_parent = q_near
        best_cost   = q_near.cost + self._dist(q_near.pos, q_new_pos)

        for node in neighbors:
            cost = node.cost + self._dist(node.pos, q_new_pos)
            if cost < best_cost and self._is_collision_free(node.pos, q_new_pos):
                best_cost   = cost
                best_parent = node

        return RRTNode(pos=q_new_pos, parent=best_parent, cost=best_cost)

    def _rewire(self, q_new, neighbors):
        """
        Check if any neighbor benefits from re-routing through q_new.
        If so, update its parent (rewire).
        This is the key asymptotic optimality step.
        """
        for node in neighbors:
            if node is q_new.parent:
                continue
            new_cost = q_new.cost + self._dist(q_new.pos, node.pos)
            if new_cost < node.cost and self._is_collision_free(q_new.pos, node.pos):
                # Detach from old parent
                if node.parent and node in node.parent.children:
                    node.parent.children.remove(node)
                # Rewire
                node.parent = q_new
                node.cost   = new_cost
                q_new.children.append(node)
                # Propagate cost update to subtree
                self._propagate_cost(node)

    def _propagate_cost(self, node):
        """Recursively update cost of all descendants after rewire."""
        for child in node.children:
            child.cost = node.cost + self._dist(node.pos, child.pos)
            self._propagate_cost(child)

    def _extract_path(self, node):
        """Trace back from goal node to root."""
        path = []
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _path_length(self, path):
        if len(path) < 2:
            return 0.0
        length = sum(
            np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            for i in range(1, len(path))
        )
        return round(length * self.grid.resolution, 4)

    def _failure(self, msg, t0, nodes_explored=0, elapsed=None):
        if elapsed is None:
            elapsed = (time.perf_counter() - t0) * 1000
        print(f"[RRT*] {msg}")
        return {
            "path":           [],
            "found":          False,
            "path_length":    0,
            "nodes_explored": nodes_explored,
            "time_ms":        elapsed,
            "algorithm":      "RRT*",
        }

    def get_tree_edges(self):
        """Return list of (parent_pos, child_pos) for visualizing the tree."""
        edges = []
        for node in self.nodes:
            if node.parent:
                edges.append((node.parent.pos, node.pos))
        return edges
