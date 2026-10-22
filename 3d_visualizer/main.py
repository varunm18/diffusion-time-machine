import argparse
from pathlib import Path
from run_dust3r import DUST3R


def main():
    parser = argparse.ArgumentParser(description="3D Scene Reconstruction & Gaussian Splatting Pipeline")
    parser.add_argument("--image_dir", type=str, default="test_workspace/images",
                        help="Path to folder containing input images")
    parser.add_argument("--workspace", type=str, default="test_workspace",
                        help="Output workspace folder for 3DGS points3D.ply and transforms.json")

    parser.add_argument("--msplat", action="store_true",
                        help="Train 3D Gaussian Splatting with msplat after reconstruction")

    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('mps', 'cuda', 'cpu', or auto-detect by default)")

    parser.add_argument("--view_output", action="store_true",
                        help="View output of dust3r reconstruction in Rerun")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive 3D visualization of final output")
    args = parser.parse_args()

    # Resolve relative paths (handles running from repo root or inside 3d_visualizer)
    script_dir = Path(__file__).resolve().parent
    image_path = Path(args.image_dir)
    if not image_path.is_absolute() and not image_path.exists() and (script_dir / image_path).exists():
        image_dir = (script_dir / image_path).resolve()
    else:
        image_dir = image_path.resolve()

    workspace = Path(args.workspace).resolve()

    # 1. Initialize and run DUSt3R reconstruction (via mini-DUSt3R)
    dust3r = DUST3R(device=args.device, view_output=args.view_output)
    dust3r.run(image_dir=image_dir, output_workspace=workspace)

    # 2. If requested, train 3D Gaussians with msplat
    if args.msplat:
        from run_splat import MSplat
        splatter = MSplat()
        splatter.train(workspace_dir=workspace)

    # 3. If requested, launch interactive 3D visualizer in Rerun
    if args.visualize:
        from run_visualization import visualize
        visualize(workspace)


if __name__ == "__main__":
    main()
