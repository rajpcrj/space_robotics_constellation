# Space Robotics — Satellite Constellation Design, Optimization & Spacecraft Dynamics

**Author:** Raj Priyadarshi
**Stack:** Python · PyTorch (CUDA) · PuLP/CBC · NumPy · OpenCV · scikit-learn · NVIDIA Isaac Sim / USD
**Domain:** orbital mechanics · rigid-body spacecraft dynamics · mixed-integer optimization · GPU-accelerated simulation · computer vision · 3D visualization

---

## Overview

A complete, multi-stage pipeline for designing and analyzing a satellite constellation that beams power to ground-based **rectennas** (rectifying antennas), built across an analytical math layer, a GPU-accelerated simulation backend, a mixed-integer optimization stage, and an Isaac Sim 3D visualization frontend.

The work spans two connected project tracks:

1. **Spacecraft dynamics & CubeSat constellation** — first-principles rigid-body physics and an orbital-geometry parametrization for building 3D constellations.
2. **Satellite-to-rectenna wireless power transfer (WPT)** — orbit propagation, satellite↔rectenna visibility analysis, constellation optimization, and animated multi-hop power paths.

Together they exercise orbital mechanics, rigid-body dynamics, integer optimization, GPU computing, graph-based routing, and computer vision inside one coherent system.

---

## Track 1 — Spacecraft Dynamics & CubeSat Constellation

### 12-DOF rigid-body dynamics (`spacecraft_dynamics.py`)

A spacecraft dynamics module implementing the full 12-degree-of-freedom rigid-body model in **two independent formulations**, used to validate each other:

- **State vector (12):** position `[r_x, r_y, r_z]`, orientation `[φ, θ, ψ]` (ZYX Euler), linear velocity `[v_x, v_y, v_z]`, angular velocity `[ω_x, ω_y, ω_z]`.
- **Control (6):** world-frame force `[f(3)]` and body-frame torque `[τ(3)]`.

**Newton–Euler formulation** (`dxdt_newton`):
```
ṙ = v
Θ̇ = Binv_zyx(φ,θ) · ω
v̇ = (f_g + f_ext) / m
ω̇ = I⁻¹ · ( −ω × (I·ω) + τ_ext )     # includes the gyroscopic term
```

**Lagrangian / generalized-coordinate formulation** (`dxdt_lagrange`), in Euler-rate coordinates:
```
J = Bᵀ I B                              # generalized inertia
Θ̈ = J⁻¹ ( Bᵀ τ_body − Ċ )              # with Coriolis/centrifugal coupling
```

**Supporting machinery:**
- Rotation utilities: `R_from_euler_zyx`, `euler_zyx_from_R`.
- Euler-rate ↔ body angular-velocity kinematic maps: `B_zyx`, `Binv_zyx` (the `B` / `B⁻¹` matrices).
- Newtonian point-mass gravity: `f_g = G·M_earth·m·r_rel / ‖r_rel‖³`.
- **Nondimensionalization** (`make_nd_scales`): characteristic scales `L0, T0 = √(L0³/μ), V0, M0, F0` for numerically well-conditioned integration at orbital scales.

**Cross-validation against MuJoCo:** both formulations are checked against an independent MuJoCo integration of the same system (stored reference trajectories `xs_mujoco` vs `xs_newton` / `xs_lagrange`), providing a third-party verification of the hand-derived dynamics.

Representative parameters (from `test_spacecraft.h5`): spacecraft mass 1600 kg, inertia diag(7429.33, 7429.33, 4608) kg·m², integration `dt = 0.05 s`, ~76,800 timesteps, initial radius ≈ 6371.5 km at ≈ 7.67 km/s.

### "Super R matrix" orbital parametrization

Each satellite's world position is composed from three ZYX-Euler rotation matrices, one per classical orbital element:
```
R_inclination = Rx(inclination)     # tilt of the orbital plane
R_raan        = Rz(raan)            # right ascension of ascending node
R_true_anomaly= Rz(true_anomaly)    # position along the orbit
rotation_matrix = R_raan @ R_true_anomaly @ R_inclination
position_world  = rotation_matrix @ position_base
```

A hierarchical class structure builds full 3D constellations by sweeping angle lists:
```
fullConstellation                    (loops inclinations)
 └─ inclinedConstellations           (one inclination; loops RAAN)
     └─ raacConstellation_beltETVs   (one RAAN; loops true anomaly)
         └─ EarthTransmissionVehicle (one (incl, RAAN, anomaly) → 1 satellite prim)
```

Each satellite is authored as a USD prim with physics (`UsdPhysics.RigidBodyAPI`/`CollisionAPI`) and animated through cooperative `asyncio` coroutines that yield to the Isaac Sim render thread, keeping the UI responsive during large-constellation playback.

---

## Track 2 — Satellite-to-Rectenna Wireless Power Transfer

### Pipeline

```
Earth map ──► rectenna placement (CV) ──► final_result.csv (rectenna lat/lon)
                                                │
              orbit grid (inclination × RAAN)   │
                        │                        │
                        ▼                        ▼
              GPU visibility simulation (PyTorch/CUDA)
                        │
                        ▼
              MILP constellation selection (PuLP/CBC)
                        │
                        ▼
              multi-hop / relay path computation
                        │
                        ▼
              Isaac Sim USD visualization (animated beams)
```

### Orbital propagation (PyTorch)

Satellite orbits are generated analytically from inclination and RAAN:
```
u = I·cos(Ω) + J·sin(Ω)                    # ascending-node direction
f = K × u                                  # in-plane perpendicular
v = Rodrigues(f, axis=u, angle=inclination)# tilt into the inclined plane
ω = √(G·M_earth / r³)                       # Kepler mean motion
p(t) = center + r·( cos(ωt+φ)·u + sin(ωt+φ)·v )
```

Rectennas are placed in an Earth-fixed (ECEF) frame from latitude/longitude and **co-rotate with the Earth** (`ω_e = 2π/86400`), with a local east/north tangent basis.

### GPU-accelerated visibility engine

The core simulation evaluates, for every (satellite, rectenna, time) triple, whether the satellite is "visible" to the rectenna via an angle-threshold test:
```
visible  iff  (sat→rect)·n̂ > 0  AND  cos(angle) > cos(threshold)   # threshold 3°–10°
```

Engineered for scale on the GPU:
- **Fully vectorized** ray-similarity over the `(N_sats × N_rect)` grid using PyTorch tensors on CUDA.
- **Time-chunked** evaluation (`chunk_size` steps at a time) to bound GPU memory while sweeping long horizons (multi-day windows at second-level `dt`).
- Outputs `sat_positions (N_sats, T, 3)`, `rect_positions (N_rect, T, 3)`, a boolean `visibility_mask (N_sats, N_rect, T)`, and per-satellite / per-rectenna visibility fractions, serialized to `.pt`.

Representative scale: **1183 candidate orbital configurations**, **50 rectennas**, 500 km orbit altitude, multi-day horizons — visibility masks of shape `(1183, 50, T)`.

### MILP constellation optimization (PuLP / CBC)

A **mixed-integer linear program** selects a minimal set of orbital configurations such that every rectenna meets a coverage threshold:

- **Decision variables:** binary `x_s ∈ {0,1}` over all grid configurations (a 100×100 inclination×RAAN grid = 10,000 candidates).
- **Objective:** `minimize Σ x_s` — fewest satellites selected.
- **Constraints:** for each rectenna *r*, `Σ_s coverage[s,r]·x_s ≥ threshold[r]` (e.g. ≥ 0.85 / 85% coverage).
- **Solver:** CBC via `PULP_CBC_CMD`, with a configurable relative MIP gap and optional time limit.
- **Independent verification:** a separate `verify_solution` routine re-checks that the selected configurations satisfy every per-rectenna coverage constraint, reporting per-rectenna margins — a built-in cross-check of the optimizer's output.

### Path computation & routing

Given the time-varying visibility mask, the pipeline computes source-to-rectenna power-delivery paths over all timesteps:
- A graph-based builder (NetworkX) constructs per-timestep connectivity (source ↔ satellites ↔ rectennas) for reference/experimentation.
- A **fast vectorized path computation** (`compute_all_paths_from_source_fast`) selects, per rectenna per timestep, the lowest-cost relay through the visible satellite set (`cost = ‖S − sat‖ + ‖sat − rect‖`), producing a `paths_matrix[rect][t]` of 3D path coordinates (or `Null` when a rectenna is inactive).

### Rectenna placement — computer vision pipeline

Rectenna ground locations are derived from an Earth land/ocean map through a self-contained CV pipeline (`EARTH_Divider.ipynb`):
1. Averaging-filter blur (50×50 kernel, `cv2.filter2D`).
2. Grayscale + binary threshold (150) → land/ocean mask.
3. Resize and region masking (exclude polar ice).
4. **K-Means** (k=50) over land pixels → cluster centers.
5. **Voronoi** tessellation to partition land into service regions.
6. Equirectangular **pixel → lat/lon** mapping.
7. Plotly 3D Earth sphere for verification.

Output: 50 rectenna coordinates driving the rest of the pipeline.

### Isaac Sim visualization frontend

A USD-based 3D frontend renders and animates the full constellation:
- Pure-USD mesh cuboids for satellites (red), rectennas (blue), and power beams (green), oriented via quaternions (`quaternion_from_z_axis`, Rodrigues).
- Asynchronous animators (`run_synchronized_animation_async`) drive satellites, rectennas, and beams together, with frame batching (e.g. up to 2000 sats / 500 rects / 200 segments per UI tick) to keep rendering smooth.
- Multi-hop/relay paths are visualized as animated beam geometry between source, relay satellites, and target rectennas.

---

## Technologies & techniques demonstrated

| Area | Techniques |
|---|---|
| **Orbital mechanics** | Kepler mean motion, inclined-orbit construction, Rodrigues rotation, ECEF/ECI geometry, parametric circular-orbit propagation |
| **Rigid-body dynamics** | Newton–Euler & Lagrangian formulations, Euler-rate kinematic maps, gyroscopic coupling, generalized inertia, nondimensionalization, MuJoCo cross-validation |
| **Optimization** | Mixed-integer linear programming (PuLP/CBC), binary set-selection under coverage constraints, independent solution verification |
| **GPU computing** | PyTorch/CUDA vectorization, time-chunked memory management, large-grid batch simulation |
| **Computer vision** | Filtering, thresholding, K-Means, Voronoi segmentation, geometric coordinate mapping |
| **Graph / routing** | Per-timestep connectivity graphs, vectorized relay path selection |
| **3D simulation / viz** | NVIDIA Isaac Sim, USD prim authoring, physics APIs, async animation |

---

## Summary

A cross-disciplinary space-robotics pipeline combining first-principles **spacecraft dynamics** (dual-formulation, MuJoCo-validated), **GPU-accelerated orbital simulation**, **mixed-integer constellation optimization** with independent verification, a **computer-vision** ground-segment placement stage, and a **3D Isaac Sim** visualization — implemented end-to-end in Python with PyTorch/CUDA and PuLP/CBC.
