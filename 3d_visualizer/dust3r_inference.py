import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo

import numpy as np
import json

def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir,
        model=model,
        device=device
    )
    log_optimized_result(optimized_results, Path("world"))

    output_dir = "test_workspace"
    export_to_3dgs_workspace(optimized_results, args.image_dir, output_dir)


def export_to_3dgs_workspace(optimized_results: OptimizedResult, original_image_dir, output_workspace_dir):
    workspace = Path(output_workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    sparse_dir = workspace / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Extract points. If point_cloud is a list of tensors, we reshape and concatenate.
    # If it is a Trimesh/Open3D object, we extract vertices.
    if isinstance(optimized_results.point_cloud, list):
        pts3d = [p.detach().cpu().numpy().reshape(-1, 3) for p in optimized_results.point_cloud]
        all_pts = np.concatenate(pts3d, axis=0)
    elif hasattr(optimized_results.point_cloud, 'vertices'): 
        all_pts = np.asarray(optimized_results.point_cloud.vertices)
    else:
        all_pts = np.asarray(optimized_results.point_cloud).reshape(-1, 3)

    # Extract colors
    if isinstance(optimized_results.rgb_hw3_list, list):
        colors = [c.detach().cpu().numpy().reshape(-1, 3) if hasattr(c, 'detach') else np.asarray(c).reshape(-1, 3) for c in optimized_results.rgb_hw3_list]
        all_colors = np.concatenate(colors, axis=0)
    elif hasattr(optimized_results.point_cloud, 'colors'):
        all_colors = np.asarray(optimized_results.point_cloud.colors)[:, :3] # trimesh colors are often RGBA
    else:
        all_colors = np.asarray(optimized_results.rgb_hw3_list).reshape(-1, 3)
        
    if all_colors.max() <= 1.0:
        all_colors = (all_colors * 255).astype(np.uint8)
    else:
        all_colors = all_colors.astype(np.uint8)

    # Write .ply file
    ply_path = sparse_dir / "points3D.ply"
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(all_pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(all_pts, all_colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
            
    print(f"Saved {len(all_pts)} points to {ply_path}")

    # Extract Camera Poses (world_T_cam_b44) and Intrinsics (K_b33)
    # Extract Camera Poses (world_T_cam_b44) and Intrinsics (K_b33)
    transforms = {"camera_model": "OPENCV", "frames": []}
    image_files = sorted([f.name for f in Path(original_image_dir).iterdir() if f.is_file()])
    
    poses = optimized_results.world_T_cam_b44
    intrinsics = optimized_results.K_b33
    rgb_images = optimized_results.rgb_hw3_list

    for i in range(len(image_files)):
        pose_np = poses[i].detach().cpu().numpy() if hasattr(poses[i], 'detach') else np.asarray(poses[i])
        K_np = intrinsics[i].detach().cpu().numpy() if hasattr(intrinsics[i], 'detach') else np.asarray(intrinsics[i])
        
        # Focal lengths are at K[0,0] and K[1,1]
        fl_x = float(K_np[0, 0])
        fl_y = float(K_np[1, 1])
        
        # Principal points (image center) are at K[0,2] and K[1,2]
        cx = float(K_np[0, 2])
        cy = float(K_np[1, 2])
        
        # Extract height and width from the image tensor shape (H, W, 3)
        h, w = int(rgb_images[i].shape[0]), int(rgb_images[i].shape[1])
        
        frame = {
            "file_path": f"images/{image_files[i]}",
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "transform_matrix": pose_np.tolist()
        }
        transforms["frames"].append(frame)

    transforms_path = workspace / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=4)
        
    print(f"Saved {len(transforms['frames'])} camera poses to {transforms_path}")

if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    main(args.image_dir)
    rr.script_teardown(args)