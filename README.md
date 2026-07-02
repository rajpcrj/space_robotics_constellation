# Satellite Constellation Design for Space-Based Wireless Power Transfer

A complete constellation-design pipeline built for **ROB-GY 7863 (Space Robotics), NYU
Tandon School of Engineering**. It answers a concrete engineering question: where should
ground stations be placed, which orbits should satellites fly, and what is the **minimum**
number of satellites that keeps every station continuously powered?

## Overview

The system places 50 ground **rectennas** (rectifying antennas) by segmenting an Earth
land/ocean map with K-Means and spherical Voronoi partitioning, propagates a grid of
candidate orbits, computes satellite-to-rectenna visibility **fully vectorized on the GPU**,
and then selects the smallest viable constellation with a **mixed-integer linear program**.
A from-scratch 12-DOF spacecraft dynamics model underpins the physics, and the whole
solution is visualized in **NVIDIA Isaac Sim** with animated multi-hop power beams.

## Technical Highlights

- **Data-driven station placement** — 50 rectennas positioned via K-Means + spherical
  Voronoi segmentation of a global land/ocean map.
- **GPU-vectorized visibility** — satellite-to-rectenna access computed across a
  (1183 × 50 × time) tensor entirely on the GPU with PyTorch/CUDA.
- **Optimal sizing** — a mixed-integer linear program (PuLP/CBC) selects the minimum
  constellation subject to per-station coverage constraints.
- **Independent verification** — a separate verifier re-checks every coverage constraint.
- **First-principles dynamics** — a full 12-DOF rigid-body spacecraft model in both
  Newton–Euler and Lagrangian formulations, cross-validated against MuJoCo.
- **Immersive visualization** — the constellation and animated multi-hop power beams
  rendered in NVIDIA Isaac Sim.

## Technologies

| Area | Details |
|------|---------|
| Language | Python |
| Numerical / GPU | PyTorch (CUDA), NumPy |
| Optimization | Mixed-Integer Linear Programming (PuLP / CBC) |
| Geospatial / CV | K-Means & spherical Voronoi (scikit-learn), OpenCV |
| Dynamics | Newton–Euler & Lagrangian 12-DOF modeling, MuJoCo cross-validation |
| Visualization | NVIDIA Isaac Sim (USD scene graph) |

## Figures

| File | Description |
|------|-------------|
| `figures/routing_isaac_sim.png` | Constellation and multi-hop power beams in Isaac Sim |
| `figures/earth_unsegmented.png` | Global land/ocean input map |
| `figures/earth_segmented.png` | K-Means + Voronoi segmentation into service regions |
| `figures/heatmap_selected_points.png` | Selected ground-rectenna locations |
| `figures/raan_histogram.png` | Distribution of RAAN across candidate orbits |
| `figures/theta_histogram.png` | Distribution of orbital phase across the candidate grid |

See [`docs/Space_Robotics_Report.pdf`](docs/Space_Robotics_Report.pdf) for the full report
and [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) for detailed design notes.

## Author

**Raj Priyadarshi** — Space Robotics, NYU Tandon School of Engineering.
