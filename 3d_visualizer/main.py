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
                        help="Device to use ('mps', 'cuda', 'cpu', or auto-detect)")
    parser.add_argument("--view_output", action="store_true",
                        help="View output of reconstruction in Rerun")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive 3D visualization after training")
    args = parser.parse_args()

    # Resolve relative paths from script directory
    script_dir = Path(__file__).resolve().parent
    image_dir = Path(args.image_dir) if Path(args.image_dir).is_absolute() else script_dir / args.image_dir
    workspace = Path(args.workspace) if Path(args.workspace).is_absolute() else script_dir / args.workspace

    # 1. Initialize and run DUSt3R reconstruction (via mini-DUSt3R)
    dust3r = DUST3R(device=args.device, view_output=args.view_output)
    dust3r.run(image_dir=image_dir, output_workspace=workspace)

    # 2. If requested, train 3D Gaussians with msplat
    if args.msplat:
        from run_splat import MSplat
        splatter = MSplat()
        splatter.train(workspace_dir=workspace)

    # 3. If requested, launch interactive 3D visualizer
    if args.visualize:
        from run_visualization import visualize_splats
        ply_file = workspace / "scene_3dgs.ply"
        if not ply_file.exists():
            ply_file = workspace / "sparse/0/points3D.ply"
        visualize_splats(str(ply_file))


if __name__ == "__main__":
    main()
