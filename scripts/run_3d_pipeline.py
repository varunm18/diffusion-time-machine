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
    parser.add_argument("--view_output", action="store_true",
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

    # 1. Initialize and run DUSt3R reconstruction
    dust3r = DUST3R(device=args.device, view_output=args.view_output)
    dust3r.run(image_dir=image_dir, output_workspace=workspace)

    # 2. If requested, train 3D Gaussians with msplat
    if args.msplat:
        splatter = MSplat()
        splatter.train(workspace_dir=workspace)

    # 3. If requested, launch interactive 3D visualizer in Rerun
    if args.visualize:
        visualize(workspace, opacity_mult=args.opacity_mult, scale_mult=args.scale_mult)


if __name__ == "__main__":
    main()
