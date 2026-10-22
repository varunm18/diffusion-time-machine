# 3D Scene Reconstruction & Gaussian Splatting Pipeline

An end-to-end Python pipeline to reconstruct 3D scenes from uncalibrated images, train 3D Gaussian Splats, and visualize the output in Rerun.

---

## Pipeline Overview

1. **Reconstruction & Camera Pose Estimation ([`run_dust3r.py`](run_dust3r.py)):**
   - Uses `mini-dust3r` to compute camera poses, intrinsics (`transforms.json`), and dense/sparse 3D point clouds (`points3D.ply`).
   - Runs on Apple Silicon (`mps`), CUDA, or CPU.
   - Automatically symlinks images into the workspace directory (0 bytes duplicated).
   - Optional Rerun preview with solid 3D point rendering via `--view_output`.

2. **3D Gaussian Splatting Optimization ([`run_splat.py`](run_splat.py)):**
   - Uses Apple Metal-accelerated `msplat` to train 3D Gaussians.
   - Exports standard `scene_3dgs.ply` and lightweight `scene_3dgs.splat`.

3. **Interactive 3D Visualization ([`run_visualization.py`](run_visualization.py)):**
   - High-performance 3D visualizer powered by **Rerun** with native EWA Gaussian Splatting rendering.
   - Displays 3D Gaussian splats, camera frustums, intrinsics, and input images in a shared coordinate space.
   - Includes `--opacity_mult` and `--scale_mult` controls to adjust density and eliminate transparency.

---

## Installation

Install the visualizer dependencies in your virtual environment:

```bash
pip install -r requirements.txt
```

---

## Quickstart (All-in-One Command)

Run reconstruction, 3DGS training, and launch the Rerun viewer in a single step:

```bash
python 3d_visualizer/main.py \
  --image_dir 3d_visualizer/testing/images/riverview \
  --workspace 3d_visualizer/test_ws \
  --msplat \
  --visualize
```

---

## Step-by-Step Usage

### 1. Camera Pose Estimation & Point Cloud
```bash
python 3d_visualizer/run_dust3r.py \
  --image_dir 3d_visualizer/testing/images/riverview \
  --workspace 3d_visualizer/test_ws
```
*Add `--view_output` to preview the point cloud and camera frustums in Rerun immediately after reconstruction.*

### 2. Train 3D Gaussian Splats
```bash
python 3d_visualizer/run_splat.py \
  --workspace 3d_visualizer/test_ws \
  --iterations 7000
```
*Outputs `scene_3dgs.ply` and `scene_3dgs.splat` into your workspace folder.*

### 3. Visualize in Rerun
```bash
python 3d_visualizer/run_visualization.py 3d_visualizer/test_ws
```

#### Adjusting Transparency & Splat Size
If the scene appears faint or translucent, increase the opacity multiplier:
```bash
# Make splats dense and solid (e.g., 5.0x):
python 3d_visualizer/run_visualization.py 3d_visualizer/test_ws --opacity_mult 5.0

# Slightly enlarge splats for continuous surface coverage:
python 3d_visualizer/run_visualization.py 3d_visualizer/test_ws --opacity_mult 5.0 --scale_mult 1.5
```

---

## Workspace Output Structure

Each workspace directory contains:
```
<workspace>/
├── images -> /path/to/original/images   # Symlink to avoid duplicating files
├── transforms.json                     # Camera poses and intrinsics
├── sparse/
│   └── 0/
│       └── points3D.ply                # Reconstructed point cloud
├── scene_3dgs.ply                      # Trained 3D Gaussian Splats
└── scene_3dgs.splat                    # Compact binary splat file
```

#### Adjusting Transparency & Splat Size
If the scene appears faint or translucent, increase the opacity multiplier:
```bash
# Make splats dense and solid (e.g., 5.0x):
python 3d_visualizer/run_visualization.py 3d_visualizer/test_ws --opacity_mult 5.0

# Slightly enlarge splats for continuous surface coverage:
python 3d_visualizer/run_visualization.py 3d_visualizer/test_ws --opacity_mult 5.0 --scale_mult 1.5
```

---

## Workspace Output Structure

Each workspace directory contains:
```
<workspace>/
├── images -> /path/to/original/images   # Symlink to avoid duplicating files
├── transforms.json                     # Camera poses and intrinsics
├── sparse/
│   └── 0/
│       └── points3D.ply                # Reconstructed point cloud
├── scene_3dgs.ply                      # Trained 3D Gaussian Splats
└── scene_3dgs.splat                    # Compact binary splat file
```

