"""
pid_controller.py
-----------------
PID-based waypoint tracking controller for a simulated drone.

Physics model: simplified point-mass with velocity damping.
Controller:    Independent PID on each axis (X, Y, Z).

PID formula:
    u(t) = Kp * e(t) + Ki * ∫e(t)dt + Kd * de/dt

Where:
    e(t)  = setpoint - current_value  (error)
    Kp    = proportional gain
    Ki    = integral gain
    Kd    = derivative gain

Author: Your Name
"""

import numpy as np
import time


# ── Single-axis PID ────────────────────────────────────────────────────────────

class PIDController:
    """
    A single-axis PID controller with anti-windup and output clamping.
    """

    def __init__(self, Kp=1.2, Ki=0.01, Kd=0.4,
                 output_min=-5.0, output_max=5.0,
                 integral_limit=10.0):
        """
        Args:
            Kp, Ki, Kd:     PID gains
            output_min/max: clamp control output (prevents actuator saturation)
            integral_limit: anti-windup clamping on integral term
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min   = output_min
        self.output_max   = output_max
        self.integral_limit = integral_limit

        self._integral  = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def compute(self, setpoint, measured, dt):
        """
        Compute PID output.

        Args:
            setpoint: desired value
            measured: current sensor value
            dt:       time step (seconds)

        Returns:
            control output (float)
        """
        error = setpoint - measured

        # Integral with anti-windup
        self._integral += error * dt
        self._integral  = np.clip(self._integral,
                                  -self.integral_limit,
                                   self.integral_limit)

        # Derivative (skip on first call to avoid spike)
        if self._first_call:
            derivative = 0.0
            self._first_call = False
        else:
            derivative = (error - self._prev_error) / dt if dt > 0 else 0.0

        self._prev_error = error

        output = (self.Kp * error +
                  self.Ki * self._integral +
                  self.Kd * derivative)

        return float(np.clip(output, self.output_min, self.output_max))

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._first_call = True


# ── 3-axis PID Drone Controller ────────────────────────────────────────────────

class DroneController:
    """
    3-axis PID waypoint tracker for a simulated drone.
    Each axis (X, Y, Z) has its own independent PID controller.
    """

    def __init__(self,
                 Kp=(1.2, 1.2, 1.5),
                 Ki=(0.01, 0.01, 0.02),
                 Kd=(0.4, 0.4, 0.5),
                 waypoint_threshold=0.8):
        """
        Args:
            Kp, Ki, Kd:           gains per axis (x, y, z)
            waypoint_threshold:   distance (voxels) to consider waypoint reached
        """
        self.pid_x = PIDController(Kp[0], Ki[0], Kd[0])
        self.pid_y = PIDController(Kp[1], Ki[1], Kd[1])
        self.pid_z = PIDController(Kp[2], Ki[2], Kd[2])
        self.waypoint_threshold = waypoint_threshold

    def compute_control(self, target, position, dt):
        """
        Compute 3D acceleration command toward target.

        Returns:
            (ax, ay, az): acceleration vector
        """
        ax = self.pid_x.compute(target[0], position[0], dt)
        ay = self.pid_y.compute(target[1], position[1], dt)
        az = self.pid_z.compute(target[2], position[2], dt)
        return np.array([ax, ay, az])

    def reached_waypoint(self, target, position):
        dist = np.linalg.norm(np.array(target) - np.array(position))
        return dist < self.waypoint_threshold

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()


# ── Drone Physics Model ────────────────────────────────────────────────────────

class DroneSimulator:
    """
    Simplified point-mass drone physics.

    State: position (x,y,z), velocity (vx,vy,vz)
    Control input: acceleration command from PID controller

    Equations of motion:
        v(t+dt) = v(t) + a(t)*dt - damping*v(t)*dt
        p(t+dt) = p(t) + v(t)*dt
    """

    def __init__(self, start_pos,
                 mass=1.0, damping=0.85, max_speed=3.0, dt=0.05):
        """
        Args:
            start_pos: initial (x,y,z)
            mass:      drone mass (kg) — scales acceleration
            damping:   velocity damping coefficient (0–1)
            max_speed: maximum speed (voxels/s)
            dt:        simulation time step (seconds)
        """
        self.pos      = np.array(start_pos, dtype=float)
        self.vel      = np.zeros(3)
        self.mass     = mass
        self.damping  = damping
        self.max_speed = max_speed
        self.dt       = dt

        self.trajectory = [tuple(self.pos.astype(int))]
        self.time_log   = [0.0]
        self.t          = 0.0

    def step(self, acceleration):
        """
        Advance simulation by one time step.

        Args:
            acceleration: np.array [ax, ay, az]
        """
        # Newton's 2nd law + velocity damping
        self.vel += (acceleration / self.mass) * self.dt
        self.vel *= (1.0 - self.damping * self.dt)

        # Clamp to max speed
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed

        # Update position
        self.pos += self.vel * self.dt
        self.t   += self.dt

        # Record integer voxel position
        int_pos = tuple(np.round(self.pos).astype(int))
        if not self.trajectory or int_pos != self.trajectory[-1]:
            self.trajectory.append(int_pos)
        self.time_log.append(self.t)

    @property
    def position(self):
        return self.pos.copy()

    def reset(self, start_pos):
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.zeros(3)
        self.trajectory = [tuple(self.pos.astype(int))]
        self.time_log   = [0.0]
        self.t          = 0.0


# ── Full flight simulation ─────────────────────────────────────────────────────

class FlightSimulator:
    """
    Integrates DroneSimulator + DroneController to fly a planned path.
    """

    def __init__(self, grid,
                 Kp=(1.2, 1.2, 1.5),
                 Ki=(0.01, 0.01, 0.02),
                 Kd=(0.4, 0.4, 0.5),
                 dt=0.05,
                 max_steps_per_waypoint=200):
        """
        Args:
            grid:                    OccupancyGrid3D (for bounds checking)
            max_steps_per_waypoint:  timeout per waypoint (prevents infinite loop)
        """
        self.grid = grid
        self.controller = DroneController(Kp, Ki, Kd)
        self.dt = dt
        self.max_steps_per_waypoint = max_steps_per_waypoint

    def fly(self, path, verbose=True):
        """
        Simulate drone flying through a planned path.

        Args:
            path: list of (x,y,z) waypoints

        Returns:
            FlightResult dict
        """
        if not path:
            return {"trajectory": [], "success": False, "reason": "Empty path"}

        drone = DroneSimulator(start_pos=path[0], dt=self.dt)
        self.controller.reset()

        total_steps  = 0
        skipped_wps  = 0
        wp_times     = []

        for wp_idx, waypoint in enumerate(path):
            wp_start_t = drone.t
            wp_steps   = 0

            while not self.controller.reached_waypoint(waypoint, drone.position):
                accel = self.controller.compute_control(
                    target=np.array(waypoint, dtype=float),
                    position=drone.position,
                    dt=self.dt
                )
                drone.step(accel)
                wp_steps   += 1
                total_steps += 1

                if wp_steps >= self.max_steps_per_waypoint:
                    if verbose:
                        print(f"[Flight] Timeout at waypoint {wp_idx} {waypoint}")
                    skipped_wps += 1
                    break

            wp_times.append(drone.t - wp_start_t)
            if verbose and wp_idx % max(1, len(path)//5) == 0:
                dist_to_goal = np.linalg.norm(
                    np.array(path[-1]) - drone.position)
                print(f"  [WP {wp_idx+1:3d}/{len(path)}] "
                      f"pos={tuple(np.round(drone.position).astype(int))}  "
                      f"dist_to_goal={dist_to_goal:.2f}")

        final_pos    = drone.position
        goal_pos     = np.array(path[-1], dtype=float)
        final_error  = float(np.linalg.norm(final_pos - goal_pos))
        success      = final_error < 2.0

        return {
            "trajectory":      drone.trajectory,
            "success":         success,
            "final_error_m":   round(final_error, 3),
            "total_time_s":    round(drone.t, 2),
            "total_steps":     total_steps,
            "waypoints_hit":   len(path) - skipped_wps,
            "waypoints_total": len(path),
            "skipped_wps":     skipped_wps,
            "wp_times":        wp_times,
            "reason":          "success" if success else "goal_not_reached",
        }

    def compare_flights(self, path_astar, path_rrt, verbose=False):
        """
        Run two flight simulations and return combined results.
        """
        print("[FlightSim] Flying A* path...")
        r_astar = self.fly(path_astar, verbose=verbose)
        self.controller.reset()

        print("[FlightSim] Flying RRT* path...")
        r_rrt   = self.fly(path_rrt, verbose=verbose)

        return {"A*": r_astar, "RRT*": r_rrt}
