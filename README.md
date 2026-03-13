# 🚁 3D Autonomous Drone Path Planning

> **A* · RRT* · Dynamic Replanning · PID Waypoint Control**  
> Built from scratch in Python — no robotics libraries used for core algorithms.

---

## 📌 Project Overview

This project implements a complete 3D drone navigation pipeline:

1. **3D Occupancy Grid** — discrete voxel environment with random/custom obstacles
2. **Path Planning** — A* (optimal grid search) and RRT* (probabilistic sampling), implemented from scratch
3. **Dynamic Replanning** — mid-flight obstacle injection triggers real-time replan
4. **PID Flight Simulation** — simulated drone tracks planned waypoints using PID control

**Goal:** Demonstrate understanding of robotics fundamentals for an MSc Robotics application.

---

## 📂 Project Structure

```
drone_path_planning/
│
├── main.py                   # Entry point — runs all phases
│
├── src/
│   ├── environment.py        # 3D Occupancy Grid (Phase 1)
│   ├── visualizer.py         # 3D rendering + animation (Phase 1)
│   ├── astar.py              # A* algorithm from scratch (Phase 2)
│   ├── rrt_star.py           # RRT* algorithm from scratch (Phase 2)
│   ├── replanner.py          # Dynamic obstacle + replanning (Phase 3)
│   └── pid_controller.py     # PID controller + drone physics (Phase 4)
│
├── results/                  # Generated plots, GIFs, saved grids
├── maps/                     # Custom map configs (.npy)
├── tests/                    # Unit tests
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# Clone the repo
git clone https://github.com/yourusername/drone-path-planning.git
cd drone-path-planning

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (all 4 phases)
python main.py

# Run with specific options
python main.py --algo astar           # A* only
python main.py --algo rrt             # RRT* only
python main.py --algo compare         # Side-by-side comparison
python main.py --seed 7               # Different random environment
python main.py --no-anim --no-show    # Headless mode (save plots only)
```

---

## 🧠 Algorithm Details

### Phase 1 — 3D Occupancy Grid

The environment is a discrete 3D voxel grid of shape `(X, Y, Z)`.

- `grid[x][y][z] = 0` → free space
- `grid[x][y][z] = 1` → obstacle

Supports box obstacles, cylinder obstacles, and random generation with fixed seeds for reproducibility.

---

### Phase 2 — Path Planning

#### A* Search

A* finds the **optimal** (shortest) path by maintaining a priority queue ordered by:

```
f(n) = g(n) + h(n)
```

Where:
- `g(n)` = actual cost from start to node `n`
- `h(n)` = heuristic estimate to goal (Euclidean distance)
- `f(n)` = estimated total cost through `n`

**Heuristic used:** Euclidean distance in 3D:

```
h(n) = sqrt((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
```

**Why Euclidean?** It is admissible (never overestimates) and consistent in 3D continuous space, guaranteeing the optimal path.

**26-connectivity:** Each voxel has up to 26 neighbors (face + edge + corner), with diagonal step costs computed as `sqrt(dx²+dy²+dz²)`.

#### RRT* (Rapidly-exploring Random Trees — Optimal)

RRT* is a **probabilistic sampling** algorithm that builds a tree by randomly exploring the space:

1. **Sample** a random point `q_rand` in the 3D grid
2. **Find nearest** tree node `q_near`
3. **Steer** from `q_near` toward `q_rand` by `step_size`
4. **Collision check** the new segment
5. **Choose best parent** — among nearby nodes, pick the one giving lowest cost to `q_new`
6. **Add node** to tree
7. **Rewire** — check if any neighbor benefits from re-routing through `q_new`

The **rewiring step** is what makes RRT\* asymptotically optimal (unlike basic RRT).

**Key difference from A\*:**

| Property | A\* | RRT\* |
|---|---|---|
| Optimality | Globally optimal | Asymptotically optimal |
| Environment | Grid-based | Continuous C-space |
| Speed | Fast (small grids) | Slower (sampling) |
| Path quality | Optimal | Improves with iterations |
| Memory | High (explored nodes) | Lower |

---

### Phase 3 — Dynamic Replanning

Simulates a **mid-flight obstacle appearance**:

1. Drone begins flying the initial planned path
2. At 40% of the path, a new obstacle is injected
3. System detects path is blocked (`path_is_clear()` check)
4. Replanning triggered from **current drone position** to goal
5. Drone continues on new path

This mimics real-world scenarios: another drone enters the airspace, a human crosses the path, a door closes.

---

### Phase 4 — PID Waypoint Tracking

The drone follows planned waypoints using a **3-axis PID controller**:

```
u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de/dt
```

Each axis (X, Y, Z) has an independent PID. The drone physics use a simplified point-mass model:

```
v(t+dt) = v(t) + a(t)·dt - damping·v(t)·dt
p(t+dt) = p(t) + v(t)·dt
```

Anti-windup and output clamping prevent actuator saturation.

---

## 📊 Sample Results

| Metric | A\* | RRT\* |
|---|---|---|
| Path Length | ~32.4 m | ~35.1 m |
| Nodes Explored | ~1,840 | ~2,100 |
| Planning Time | ~12 ms | ~180 ms |
| Replan Time | ~9 ms | ~210 ms |

> Results vary by environment seed. Run `python main.py --algo compare` to generate your own metrics.

---

## 🔬 Design Decisions & Tradeoffs

**Why implement from scratch?**
Using libraries like `python-robotics` hides the actual algorithm. Implementing A* and RRT* manually forces understanding of the data structures (min-heap, tree), the math (heuristic admissibility, cost propagation), and the edge cases (out-of-bounds, start/goal in obstacle).

**Why 26-connectivity over 6?**
In 3D space, restricting to only face-adjacent neighbors forces the drone to take "staircase" paths instead of diagonal shortcuts. 26-connectivity produces more natural, shorter paths at the cost of a larger branching factor.

**Why Euclidean heuristic over Manhattan?**
Manhattan distance is inadmissible in 26-connected 3D grids (it underestimates less, but counts only axis-aligned moves). Euclidean distance is strictly admissible and consistent, guaranteeing A* finds the optimal path.

---

## 🔧 Dependencies

| Library | Version | Usage |
|---|---|---|
| `numpy` | ≥1.24 | Grid, vector math, path operations |
| `matplotlib` | ≥3.7 | 3D visualization, animation |

No ROS, no AirSim, no robotics libraries. Pure Python.

---

## 📈 Possible Extensions

- [ ] SLAM integration (build map while navigating)
- [ ] RL-based planner (replace A* with trained policy)
- [ ] Multi-drone coordination (shared occupancy grid)
- [ ] Real hardware deployment (ArduPilot / DJI Tello)
- [ ] Streamlit web UI for mission control demo

---

## 👤 Author

**Your Name**  
B.Tech Computer Science & Engineering  
Applying for MSc Robotics — TU Munich / KIT / TU Berlin  

[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📄 License

MIT License — free to use, modify, and distribute.
