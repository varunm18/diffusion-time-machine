import json
import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.api import inferece_dust3r, log_optimized_result


def save_transforms(frames, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"camera_model": "OPENCV", "frames": frames}, f, indent=4)
    print(f"Saved {len(frames)} camera frames to {output_path}")


class DUST3R:
    def __init__(self, device=None, model_name=None, view_output=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[DUSt3R] Initializing mini-DUSt3R on device='{self.device}'")

        ckpt = model_name or "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        self.model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(self.device)
        self.view_output = view_output

    def run(self, image_dir, output_workspace="test_workspace"):
        image_dir = Path(image_dir).resolve()
        workspace = Path(output_workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        # Create or update symlink to shared images folder (0 bytes copied)
        images_link = workspace / "images"
        if images_link.is_symlink() and images_link.resolve() != image_dir:
            images_link.unlink()
        if not images_link.exists() and image_dir != images_link:
            images_link.symlink_to(image_dir, target_is_directory=True)
            print(f"[DUSt3R] Linked shared images: {images_link} -> {image_dir}")

        print(f"[DUSt3R] Running mini-DUSt3R inference on {len(image_files)} images from {image_dir}...")
        results = inferece_dust3r(image_dir, model=self.model, device=self.device)

        # 1. Save point cloud (results.point_cloud is a trimesh.PointCloud)
        ply_path = workspace / "sparse" / "0" / "points3D.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        results.point_cloud.export(str(ply_path))
        print(f"[DUSt3R] Saved {len(results.point_cloud.vertices):,} points to {ply_path}")

        # 2. Save camera poses & intrinsics
        frames = []
        for i, img_file in enumerate(image_files):
            pose = results.world_T_cam_b44[i]
            K = results.K_b33[i]
            h, w = results.rgb_hw3_list[i].shape[:2]
            frames.append({
                "file_path": f"images/{img_file.name}",
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "w": int(w),
                "h": int(h),
                "transform_matrix": pose.tolist(),
            })

        save_transforms(frames, workspace / "transforms.json")

        # 3. Optional viewer
        if self.view_output:
            import rerun as rr
            print("[DUSt3R] Spawning Rerun 3D viewer...")
            rr.init("mini-dust3r", spawn=True)
            log_optimized_result(results, Path("world"))

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DUSt3R 3D Reconstruction Pipeline (via mini-DUSt3R)")
    parser.add_argument("--image_dir", type=str, default="test_workspace/images",
                        help="Path to folder containing input images")
    parser.add_argument("--workspace", type=str, default="test_workspace",
                        help="Output workspace folder for 3DGS points3D.ply and transforms.json")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('mps', 'cuda', 'cpu', or auto-detect)")
    parser.add_argument("--view_output", action="store_true",
                        help="View output in Rerun 3D viewer")
    args = parser.parse_args()

    dust3r = DUST3R(device=args.device, view_output=args.view_output)
    dust3r.run(image_dir=args.image_dir, output_workspace=args.workspace)