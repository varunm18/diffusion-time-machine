import sys
import json
from pathlib import Path
import numpy as np
import torch
import trimesh
import rerun as rr

# Ensure dust3r submodule can be imported
DUST3R_ROOT = Path(__file__).resolve().parent / "submodules" / "dust3r"
if DUST3R_ROOT.exists():
    sys.path.append(str(DUST3R_ROOT))
    if (DUST3R_ROOT / "croco").exists():
        sys.path.append(str(DUST3R_ROOT / "croco"))


def save_transforms(frames, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"camera_model": "OPENCV", "frames": frames}, f, indent=4)
    print(f"Saved {len(frames)} camera frames to {output_path}")


class DUST3R:
    def __init__(self, mode="dust3r", device=None, model_name=None, view_output=False):
        self.mode = mode.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[DUSt3R] Initializing mode='{self.mode}' on device='{self.device}'")

        if self.mode == "mini_dust3r":
            from mini_dust3r.model import AsymmetricCroCo3DStereo
            ckpt = model_name or "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        else:
            from dust3r.model import AsymmetricCroCo3DStereo
            ckpt = model_name or "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

        self.model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(self.device)
        self.view_output = view_output

    def run(self, image_dir, output_workspace="test_workspace", niter=300):
        image_dir = Path(image_dir)
        workspace = Path(output_workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        if self.mode == "mini_dust3r":
            return self._run_mini(image_dir, workspace)
        else:
            return self._run_dust3r(image_dir, workspace, niter=niter)

    def _run_mini(self, image_dir, workspace):
        from mini_dust3r.api import inferece_dust3r

        print(f"[mini_dust3r] Running inference on {image_dir}...")
        results = inferece_dust3r(image_dir, model=self.model, device=self.device)

        # 1. Save point cloud (mini_dust3r result is already a trimesh.PointCloud)
        ply_path = workspace / "sparse" / "0" / "points3D.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        results.point_cloud.export(str(ply_path))
        print(f"Saved {len(results.point_cloud.vertices)} points to {ply_path}")

        # 2. Save camera poses & intrinsics
        image_files = sorted([f.name for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        frames = []
        for i, img_name in enumerate(image_files):
            pose = results.world_T_cam_b44[i]
            K = results.K_b33[i]
            h, w = results.rgb_hw3_list[i].shape[:2]
            frames.append({
                "file_path": f"images/{img_name}",
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "w": int(w),
                "h": int(h),
                "transform_matrix": pose.tolist(),
            })

        save_transforms(frames, workspace / "transforms.json")
        if self.view_output:
            import rerun as rr
            from mini_dust3r.api import log_optimized_result
            print("[mini_dust3r] Spawning Rerun 3D viewer...")
            rr.init("mini-dust3r", spawn=True)
            log_optimized_result(results, Path("world"))

        return results

    def _run_dust3r(self, image_dir, workspace, niter=300):
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        image_files = sorted([str(f) for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        print(f"[dust3r] Loading {len(image_files)} images...")
        images = load_images(image_files, size=512)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

        print("[dust3r] Running pairwise inference...")
        output = inference(pairs, self.model, self.device, batch_size=1)

        print("[dust3r] Running global alignment...")
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

        # 1. Save point cloud
        masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
        pts3d = [p.detach().cpu().numpy()[m] for p, m in zip(scene.get_pts3d(), masks)]
        colors = [img[m] for img, m in zip(scene.imgs, masks)]
        all_pts = np.concatenate(pts3d, axis=0)
        all_rgb = (np.concatenate(colors, axis=0) * 255).astype(np.uint8)

        ply_path = workspace / "sparse" / "0" / "points3D.ply"
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.PointCloud(vertices=all_pts, colors=all_rgb).export(str(ply_path))
        print(f"Saved {len(all_pts)} points to {ply_path}")

        # 2. Save camera poses & intrinsics
        poses = scene.get_im_poses().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy().ravel()
        pp = scene.get_principal_points().detach().cpu().numpy()

        frames = []
        for i, img_path in enumerate(image_files):
            h, w = scene.imgs[i].shape[:2]
            frames.append({
                "file_path": f"images/{Path(img_path).name}",
                "fl_x": float(focals[i]),
                "fl_y": float(focals[i]),
                "cx": float(pp[i, 0]),
                "cy": float(pp[i, 1]),
                "w": int(w),
                "h": int(h),
                "transform_matrix": poses[i].tolist(),
            })

        save_transforms(frames, workspace / "transforms.json")

        if self.view_output:
            print("[dust3r] Opening 3D scene visualizer...")
            scene.show()

        return scene