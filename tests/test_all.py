"""
tests/test_all.py
-----------------
Unit tests for all project modules.

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.environment    import OccupancyGrid3D
from src.astar          import AStarPlanner, smooth_path
from src.rrt_star       import RRTStarPlanner
from src.pid_controller import PIDController, DroneController, FlightSimulator


# ── Environment tests ──────────────────────────────────────────────────────────

class TestOccupancyGrid:

    def test_init(self):
        g = OccupancyGrid3D(10, 10, 5)
        assert g.grid.shape == (10, 10, 5)
        assert g.grid.sum() == 0

    def test_add_obstacle(self):
        g = OccupancyGrid3D(10, 10, 5)
        g.add_obstacle(3, 3, 2)
        assert g.is_obstacle(3, 3, 2)
        assert not g.is_free(3, 3, 2)

    def test_bounds_check(self):
        g = OccupancyGrid3D(5, 5, 5)
        assert not g._in_bounds(-1, 0, 0)
        assert not g._in_bounds(5, 0, 0)
        assert g._in_bounds(4, 4, 4)

    def test_box_obstacle(self):
        g = OccupancyGrid3D(10, 10, 10)
        g.add_box_obstacle(2, 2, 2, 4, 4, 4)
        assert g.is_obstacle(2, 2, 2)
        assert g.is_obstacle(3, 3, 3)
        assert g.is_free(5, 5, 5)

    def test_neighbors_26(self):
        g = OccupancyGrid3D(10, 10, 5)
        neighbors = g.get_neighbors(5, 5, 2, connectivity=26)
        assert len(neighbors) == 26

    def test_neighbors_6(self):
        g = OccupancyGrid3D(10, 10, 5)
        neighbors = g.get_neighbors(5, 5, 2, connectivity=6)
        assert len(neighbors) == 6

    def test_obstacle_density(self):
        g = OccupancyGrid3D(10, 10, 10)
        g.add_box_obstacle(0, 0, 0, 5, 5, 5)
        assert 0 < g.obstacle_density() < 1

    def test_path_is_clear(self):
        g = OccupancyGrid3D(10, 10, 5)
        path = [(1,1,1), (2,2,1), (3,3,1)]
        assert g.path_is_clear(path)
        g.add_obstacle(2, 2, 1)
        assert not g.path_is_clear(path)

    def test_dynamic_obstacle(self):
        g = OccupancyGrid3D(10, 10, 5)
        g.inject_dynamic_obstacle(5, 5, 2, size=1)
        assert g.is_obstacle(5, 5, 2)


# ── A* tests ───────────────────────────────────────────────────────────────────

class TestAStar:

    def make_simple_grid(self):
        g = OccupancyGrid3D(10, 10, 5)
        return g

    def test_finds_path(self):
        g = self.make_simple_grid()
        planner = AStarPlanner(g)
        result  = planner.plan((0,0,0), (9,9,4))
        assert result["found"]
        assert len(result["path"]) >= 2
        assert result["path"][0]  == (0,0,0)
        assert result["path"][-1] == (9,9,4)

    def test_no_path_blocked(self):
        g = OccupancyGrid3D(5, 5, 3)
        # Wall across the entire grid
        for y in range(5):
            for z in range(3):
                g.add_obstacle(2, y, z)
        planner = AStarPlanner(g)
        result  = planner.plan((0,0,0), (4,4,2))
        assert not result["found"]

    def test_start_in_obstacle(self):
        g = self.make_simple_grid()
        g.add_obstacle(0, 0, 0)
        planner = AStarPlanner(g)
        result  = planner.plan((0,0,0), (9,9,4))
        assert not result["found"]

    def test_path_length_positive(self):
        g = self.make_simple_grid()
        planner = AStarPlanner(g)
        result  = planner.plan((0,0,0), (5,5,2))
        assert result["path_length"] > 0

    def test_smooth_path(self):
        g = self.make_simple_grid()
        planner = AStarPlanner(g)
        result  = planner.plan((0,0,0), (9,9,4))
        raw_len = len(result["path"])
        smoothed = smooth_path(result["path"], g)
        assert len(smoothed) <= raw_len
        assert smoothed[0]  == result["path"][0]
        assert smoothed[-1] == result["path"][-1]


# ── RRT* tests ─────────────────────────────────────────────────────────────────

class TestRRTStar:

    def make_grid(self):
        g = OccupancyGrid3D(15, 15, 8)
        g.add_random_obstacles(num_boxes=3, num_cylinders=1, seed=1)
        g.grid[0, 0, 0] = 0
        g.grid[14, 14, 7] = 0
        return g

    def test_finds_path(self):
        g = self.make_grid()
        planner = RRTStarPlanner(g, max_iter=2000, seed=42)
        result  = planner.plan((0,0,0), (14,14,7))
        assert result["found"]
        assert result["path"][0]  == (0,0,0)
        assert result["path"][-1] == (14,14,7)

    def test_nodes_explored(self):
        g = self.make_grid()
        planner = RRTStarPlanner(g, max_iter=500, seed=0)
        result  = planner.plan((0,0,0), (14,14,7))
        assert result["nodes_explored"] <= 500

    def test_start_in_obstacle(self):
        g = OccupancyGrid3D(10, 10, 5)
        g.add_obstacle(0, 0, 0)
        planner = RRTStarPlanner(g)
        result  = planner.plan((0,0,0), (9,9,4))
        assert not result["found"]


# ── PID Controller tests ───────────────────────────────────────────────────────

class TestPID:

    def test_converges_to_setpoint(self):
        pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.3)
        value = 0.0
        dt = 0.05
        for _ in range(200):
            u     = pid.compute(setpoint=10.0, measured=value, dt=dt)
            value += u * dt
        assert abs(value - 10.0) < 1.0  # within 1 unit of setpoint

    def test_output_clamped(self):
        pid = PIDController(Kp=100.0, output_min=-5.0, output_max=5.0)
        u   = pid.compute(setpoint=1000.0, measured=0.0, dt=0.1)
        assert -5.0 <= u <= 5.0

    def test_reset_clears_state(self):
        pid = PIDController()
        pid.compute(setpoint=5.0, measured=0.0, dt=0.1)
        pid.reset()
        assert pid._integral == 0.0

    def test_drone_controller_3axis(self):
        g  = OccupancyGrid3D(20, 20, 10)
        dc = DroneController()
        accel = dc.compute_control(
            target=np.array([5.0, 5.0, 3.0]),
            position=np.array([0.0, 0.0, 0.0]),
            dt=0.05
        )
        assert len(accel) == 3
        assert all(np.isfinite(accel))

    def test_flight_simulator_runs(self):
        g       = OccupancyGrid3D(15, 15, 8)
        sim     = FlightSimulator(g, dt=0.1)
        path    = [(0,0,0), (3,3,2), (7,7,4), (10,10,6)]
        result  = sim.fly(path, verbose=False)
        assert "trajectory" in result
        assert len(result["trajectory"]) > 0


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
