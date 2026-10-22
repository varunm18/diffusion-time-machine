# 3D Visualization & Reconstruction Pipeline

This document summarizes the design, architecture, implementation decisions, and usage of the 3D visualization and reconstruction pipeline developed on the `varun` branch.

---

## 1. Overview & Goals

The Diffusion Time Machine generates historical views of scenes across time. To evaluate, verify, and ground these generations, we require a robust 3D representation:
1. **Geometric Grounding & Pose Estimation:** Accurately estimate camera poses ($c2w$ / $w2c$) and intrinsics ($K$) from unposed multi-view imagery.
2. **Dense 3D Scene Representation:** Fit continuous 3D Gaussian Splats (3DGS) to capture view-dependent radiance, geometry, and textures.
3. **Interactive 3D Inspection:** Seamlessly inspect camera frustums, pinhole projections, original photos, and dense 3D Gaussians in an interactive viewer.

---

## 2. Architecture & Data Flow

The pipeline operates in three stages:

```
                          Uncalibrated Images
                                  │
                                  ▼
     ┌────────────────────────────────────────────────────────┐
     │ Step 1: DUSt3R Reconstruction (src/viz/reconstruction) │
     │  - Pairwise stereo matching & global point optimization│
     │  - Exports camera intrinsics & poses (transforms.json) │
     │  - Exports sparse/dense point cloud (points3D.ply)     │
     └────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
     ┌────────────────────────────────────────────────────────┐
     │ Step 2: MSplat Optimization (src/viz/splat)            │
     │  - Metal GPU-accelerated 3DGS training (Apple Silicon) │
     │  - Rasterizes & optimizes 3D Gaussians from cameras    │
     │  - Exports scene_3dgs.ply and scene_3dgs.splat         │
     └────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
     ┌────────────────────────────────────────────────────────┐
     │ Step 3: Rerun 3D Visualizer (src/viz/viewer)           │
     │  - Native EWA Gaussian Splatting (GaussianSplats3D)    │
     │  - Solid surface opacity boost & scale controls        │
     │  - Camera frustums with aligned input photo overlays   │
     └────────────────────────────────────────────────────────┘
```

### Directory Structure & Artifacts

Each reconstruction workspace adheres to standard NeRF/COLMAP conventions:

```
workspace/<scene_name>/
├── images -> /path/to/source/images/     # Symlink to avoid duplicating raw image bytes
├── transforms.json                       # Camera poses, intrinsics (OPENCV RDF convention)
├── sparse/
│   └── 0/
│       └── points3D.ply                  # Initial point cloud from DUSt3R
├── scene_3dgs.ply                        # Optimized 3D Gaussian Splat scene
└── scene_3dgs.splat                      # Lightweight binary splat format
```

---

## 3. Key Technical Decisions & Challenges Solved

### A. Modular `mini-dust3r` vs. Monolithic Submodules
- **Prior state:** The project previously relied on a cloned `dust3r` git submodule with nested dependencies, requiring manual subfolder installations and fragile path hacks.
- **Resolution:** Replaced the submodule with the pip-installable [`mini-dust3r`](https://github.com/cvg/mini-dust3r) package. Cleaned `.gitmodules` and deleted the `submodules/` directory entirely.

### B. High-Performance Visualization via Rerun SDK
- **Prior state:** Early prototypes experimented with web viewers (Viser) which suffered from WebGL context loss and poor handling of dense splats on local hardware.
- **Resolution:** Integrated [`rerun-sdk`](https://rerun.io) (`>=0.40.0`). Rerun provides native Rust/Metal GPU EWA Gaussian Splatting (`rr.GaussianSplats3D`), interactive camera flythroughs, pinhole camera frustums, and side-by-side multi-view inspection.
- **Deprecation Fixes:** Migrated away from deprecated `timeless=True` to Rerun's modern `static=True` API for persistent 3D entities.
- **Session Cleanup:** Added proactive disconnection handling (`disconnect_orphaned_recordings()`) to eliminate gRPC connection crashes when reopening or relaunching viewers after closing a previous window.

### C. Dense Surface Opacity & Visual Quality
- **Problem:** Initial 3DGS checkpoints often appeared ghostly or translucent because initial sigmoid opacities start around $\sigma(-2.197) \approx 0.10$ (10%). Sparse point clouds rendered as faint single-pixel dots.
- **Resolution:**
  - Added an adjustable opacity boost (`--opacity_mult`, default `3.0x`–`5.0x`) in `src/viz/viewer.py` to render dense, solid surfaces.
  - Implemented unit quaternion normalization for all Gaussian orientations: $q / \|q\|$.
  - Extracted base RGB colors from degree-0 spherical harmonics ($C_0 = \sqrt{1 / 4\pi} \approx 0.28209$).
  - Added adaptive radius scaling for point clouds so vertices render as solid 3D spheres scaled to the scene bounding box.

### D. Apple Silicon Acceleration via `msplat`
- Utilizes `msplat` to train 3D Gaussians locally on Apple Silicon Metal compute shaders without requiring an external CUDA cluster.
- Pinned virtual environment to Python 3.12 (as `msplat` distributes precompiled Apple Silicon Metal wheels for Python 3.12/3.13).

### E. Unified Package & Requirements
- Consolidates all root dependencies into a single root [`requirements.txt`](file:///Users/varun/Documents/673/diffusion-time-machine/requirements.txt).
- Avoids duplicated or conflicting dependency files.

---

## 4. Codebase Organization

The visualization pipeline is structured as follows:

```
diffusion-time-machine/
├── src/
│   └── viz/
│       ├── __init__.py           # Public exports: DUST3R, MSplat, visualize
│       ├── reconstruction.py     # mini-DUSt3R pose estimation & transforms.json generator
│       ├── splat.py              # msplat Metal 3DGS trainer & PLY/SPLAT exporter
│       └── viewer.py             # Rerun 3D viewer (Gaussian splats, frustums, images)
├── scripts/
│   └── run_3d_pipeline.py        # CLI entry point for full pipeline execution
├── tests/
│   ├── viz/
│   │   ├── test_reconstruction.py # Unit tests for DUSt3R reconstruction
│   │   ├── test_splat.py          # Unit tests for msplat training & exports
│   │   └── test_viewer.py         # Unit tests for Rerun logging & frustums
│   └── scripts/
│       └── test_run_3d_pipeline.py# Unit tests for CLI argument parsing & workflow
└── notebooks/
    └── struct_splat_test.ipynb   # Interactive exploratory notebook
```

---

## 5. Usage Guide

### Running the End-to-End CLI

Run the full pipeline on a set of images:

```bash
python scripts/run_3d_pipeline.py \
  --image_dir data/sample_images/riverview \
  --workspace data/workspaces/riverview \
  --view_reconstruction \
  --msplat \
  --visualize
```

#### CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--image_dir` | Path to input images directory | `data/sample_images/riverview` |
| `--workspace` | Destination workspace for outputs | `workspace` |
| `--view_reconstruction` | Preview DUSt3R point cloud and poses in Rerun immediately | `False` |
| `--msplat` | Run Metal 3DGS optimization | `False` |
| `--visualize` | Launch interactive Rerun viewer after optimization | `False` |
| `--opacity_mult` | Splat opacity multiplier (boost to make surfaces solid) | `3.0` |
| `--scale_mult` | Splat radius multiplier | `1.0` |
| `--device` | Compute device (`mps`, `cuda`, `cpu`, or auto-detect) | Auto |

### Inspecting an Existing Reconstruction

To open the interactive viewer on an already computed workspace or direct `.ply` file:

```bash
# View an existing workspace folder (reads scene_3dgs.ply + transforms.json + images)
python -m src.viz.viewer --target data/workspaces/riverview

# Or view a specific standalone PLY file with custom opacity boost
python -m src.viz.viewer --target data/workspaces/riverview/scene_3dgs.ply --opacity_mult 4.0
```

### Python API

Modules can be directly imported in custom training scripts or evaluation loops:

```python
from src.viz import DUST3R, MSplat, visualize

# Step 1: Camera Pose Estimation & Point Cloud
dust3r = DUST3R(device="mps")
dust3r.run(
    image_dir="data/sample_images/riverview",
    output_workspace="data/workspaces/riverview",
)

# Step 2: 3D Gaussian Splat Optimization
splatter = MSplat(iterations=7000)
splatter.train("data/workspaces/riverview")

# Step 3: Interactive Rerun Visualization
visualize("data/workspaces/riverview", opacity_mult=3.0)
```

---

## 6. Verification & Test Suite

All unit tests are isolated and run in seconds using mocks to verify logic without downloading multi-gigabyte models or requiring GPU execution:

```bash
# Run all tests
pytest tests/

# Run visualization tests specifically
pytest tests/viz/ tests/scripts/
```

- **Current Status:** 102 passing tests across `tests/viz/`, `tests/scripts/`, and `tests/data_pipeline/`.
