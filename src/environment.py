"""
environment.py
--------------
3D Occupancy Grid for drone path planning.
Each cell: 0 = free space, 1 = obstacle

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os


class OccupancyGrid3D:
    """
    Represents a 3D environment as a discrete voxel grid.
    Shape: (x_size, y_size, z_size)
    Value: 0 = free, 1 = obstacle
    """

    def __init__(self, x_size=20, y_size=20, z_size=10, resolution=1.0):
        """
        Args:
            x_size: Number of cells along X axis
            y_size: Number of cells along Y axis
            z_size: Number of cells along Z axis
            resolution: Real-world size of each cell (meters)
        """
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.resolution = resolution
        self.grid = np.zeros((x_size, y_size, z_size), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Obstacle Placement
    # ------------------------------------------------------------------

    def add_obstacle(self, x, y, z):
        """Mark a single voxel as obstacle."""
        if self._in_bounds(x, y, z):
            self.grid[x, y, z] = 1

    def add_box_obstacle(self, x_min, y_min, z_min, x_max, y_max, z_max):
        """Fill a rectangular box region with obstacles."""
        x_min, x_max = max(0, x_min), min(self.x_size, x_max)
        y_min, y_max = max(0, y_min), min(self.y_size, y_max)
        z_min, z_max = max(0, z_min), min(self.z_size, z_max)
        self.grid[x_min:x_max, y_min:y_max, z_min:z_max] = 1

    def add_cylinder_obstacle(self, cx, cy, radius, z_min, z_max):
        """Add a vertical cylinder obstacle."""
        for x in range(self.x_size):
            for y in range(self.y_size):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist <= radius:
                    z_lo = max(0, z_min)
                    z_hi = min(self.z_size, z_max)
                    self.grid[x, y, z_lo:z_hi] = 1

    def add_random_obstacles(self, num_boxes=8, num_cylinders=4, seed=42):
        """
        Populate the grid with random box and cylinder obstacles.
        Uses a fixed seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        for _ in range(num_boxes):
            x = rng.integers(2, self.x_size - 4)
            y = rng.integers(2, self.y_size - 4)
            z = rng.integers(0, self.z_size - 2)
            w = rng.integers(1, 4)
            d = rng.integers(1, 4)
            h = rng.integers(1, 4)
            self.add_box_obstacle(x, y, z, x + w, y + d, z + h)

        for _ in range(num_cylinders):
            cx = rng.integers(3, self.x_size - 3)
            cy = rng.integers(3, self.y_size - 3)
            r = rng.integers(1, 3)
            z_top = rng.integers(3, self.z_size)
            self.add_cylinder_obstacle(cx, cy, r, 0, z_top)

    def inject_dynamic_obstacle(self, x, y, z, size=2):
        """
        Inject a new obstacle mid-simulation (used in Phase 3).
        Returns the obstacle's bounding box for logging.
        """
        self.add_box_obstacle(x, y, z, x + size, y + size, z + size)
        return (x, y, z, x + size, y + size, z + size)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_free(self, x, y, z):
        """Return True if cell is within bounds and not occupied."""
        return self._in_bounds(x, y, z) and self.grid[x, y, z] == 0

    def is_obstacle(self, x, y, z):
        return self._in_bounds(x, y, z) and self.grid[x, y, z] == 1

    def _in_bounds(self, x, y, z):
        return (0 <= x < self.x_size and
                0 <= y < self.y_size and
                0 <= z < self.z_size)

    def get_neighbors(self, x, y, z, connectivity=26):
        """
        Return free neighboring voxels.
        connectivity=6  → face neighbors only
        connectivity=26 → face + edge + corner neighbors
        """
        neighbors = []
        deltas = []

        if connectivity == 6:
            deltas = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        else:  # 26-connectivity
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        deltas.append((dx, dy, dz))

        for dx, dy, dz in deltas:
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.is_free(nx, ny, nz):
                cost = np.sqrt(dx**2 + dy**2 + dz**2)  # Euclidean step cost
                neighbors.append(((nx, ny, nz), cost))

        return neighbors

    def path_is_clear(self, path):
        """Check whether any voxel in a path list is now blocked."""
        for (x, y, z) in path:
            if not self.is_free(x, y, z):
                return False
        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        """Save the occupancy grid to a .npy file."""
        np.save(filepath, self.grid)
        print(f"[Grid] Saved to {filepath}")

    def load(self, filepath):
        """Load an occupancy grid from a .npy file."""
        self.grid = np.load(filepath)
        self.x_size, self.y_size, self.z_size = self.grid.shape
        print(f"[Grid] Loaded from {filepath} — shape {self.grid.shape}")

    def save_config(self, filepath):
        """Save grid metadata as JSON."""
        config = {
            "x_size": self.x_size,
            "y_size": self.y_size,
            "z_size": self.z_size,
            "resolution": self.resolution,
            "obstacle_count": int(np.sum(self.grid))
        }
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def obstacle_density(self):
        total = self.x_size * self.y_size * self.z_size
        filled = np.sum(self.grid)
        return filled / total

    def __repr__(self):
        return (f"OccupancyGrid3D(shape=({self.x_size},{self.y_size},{self.z_size}), "
                f"density={self.obstacle_density():.2%})")
