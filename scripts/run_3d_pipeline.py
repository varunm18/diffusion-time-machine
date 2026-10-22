"""
End-to-End 3D Scene Reconstruction and Gaussian Splatting CLI.

Orchestrates the entire 3D pipeline:
1. Reconstruction: mini-DUSt3R pose estimation & sparse point cloud (transforms.json + points3D.ply).
2. Optimization: Metal-accelerated msplat 3D Gaussian Splatting (scene_3dgs.ply + scene_3dgs.splat).
3. Visualization: Interactive Rerun 3D viewer with camera frustums and non-transparent splats.

Example usage:
    python scripts/run_3d_pipeline.py \
      --image_dir data/sample_images/riverview \
      --workspace workspace/riverview \
      --msplat \
      --visualize
"""

import sys
from pathlib import Path
import argparse

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.viz import DUST3R, MSplat, visualize


def main():
    parser = argparse.ArgumentParser(description="3D Scene Reconstruction & Gaussian Splatting Pipeline")
    parser.add_argument("--image_dir", type=str, default="data/sample_images/riverview",
                        help="Path to folder containing input images")
    parser.add_argument("--workspace", type=str, default="workspace",
                        help="Output workspace folder for 3DGS points3D.ply and transforms.json")
    parser.add_argument("--msplat", action="store_true",
                        help="Train 3D Gaussian Splatting with msplat after reconstruction")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('mps', 'cuda', 'cpu', or auto-detect by default)")
    parser.add_argument("--view_reconstruction", action="store_true",
                        help="View output of reconstruction in Rerun")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive 3D visualization after training")
    parser.add_argument("--opacity_mult", type=float, default=3.0,
                        help="Multiplier for splat opacity in viewer (default: 3.0)")
    parser.add_argument("--scale_mult", type=float, default=1.0,
                        help="Multiplier for splat scale/size in viewer (default: 1.0)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    workspace = Path(args.workspace).resolve()

    # Step 1: Camera Pose Estimation & Point Cloud Reconstruction
    print(f"\n=== Step 1: DUSt3R Reconstruction ===")
    dust3r = DUST3R(device=args.device, view_reconstruction=args.view_reconstruction)
    dust3r.run(image_dir=image_dir, output_workspace=workspace)

    # Step 2: 3D Gaussian Splatting Optimization (Metal GPU)
    if args.msplat:
        print(f"\n=== Step 2: MSplat 3DGS Training ===")
        splatter = MSplat()
        splatter.train(workspace_dir=workspace)

    # Step 3: Interactive Rerun Visualization
    if args.visualize:
        print(f"\n=== Step 3: Rerun 3D Visualization ===")
        visualize(workspace, opacity_mult=args.opacity_mult, scale_mult=args.scale_mult)


if __name__ == "__main__":
    main()
