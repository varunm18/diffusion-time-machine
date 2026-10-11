import argparse
from pathlib import Path
from run_dust3r import DUST3R

def main():
    parser = argparse.ArgumentParser(description="DUSt3R & mini-DUSt3R 3D Reconstruction Pipeline")
    parser.add_argument("--mode", type=str, default="dust3r", choices=["dust3r", "mini_dust3r"],
                        help="Backend to use: 'dust3r' (full repo) or 'mini_dust3r' (lightweight)")
    parser.add_argument("--image_dir", type=str, default="test_workspace/images",
                        help="Path to folder containing input images")
    parser.add_argument("--workspace", type=str, default="test_workspace",
                        help="Output workspace folder for 3DGS points3D.ply and transforms.json")
    parser.add_argument("--msplat", action="store_true",
                        help="Train 3D Gaussian Splatting with msplat after reconstruction")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('mps', 'cuda', 'cpu', or auto-detect)")
    parser.add_argument("--view_output", action="store_true",
                            help="View output of dust3r")
    args = parser.parse_args()

    # Resolve relative paths from script directory
    script_dir = Path(__file__).resolve().parent
    image_dir = Path(args.image_dir) if Path(args.image_dir).is_absolute() else script_dir / args.image_dir
    workspace = Path(args.workspace) if Path(args.workspace).is_absolute() else script_dir / args.workspace

    # Initialize and run
    dust3r = DUST3R(mode=args.mode, device=args.device, view_output=args.view_output)
    dust3r.run(image_dir=image_dir, output_workspace=workspace)

if __name__ == "__main__":
    main()