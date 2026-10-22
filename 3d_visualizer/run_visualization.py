import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from plyfile import PlyData
import rerun as rr

SH_C0 = 0.28209479177387814


def log_ply_file(ply_file: Path, opacity_mult: float = 3.0, scale_mult: float = 1.0):
    """Log a 3DGS PLY or point cloud PLY with controllable opacity and scale."""
    print(f"[Rerun] Reading {ply_file.name}...")
    plydata = PlyData.read(str(ply_file))
    v = plydata["vertex"]
    prop_names = [p.name for p in v.properties]

    # Check if this is a 3D Gaussian Splat PLY
    if "scale_0" in prop_names and "rot_0" in prop_names:
        print(f"[Rerun] Detected 3D Gaussian Splats ({len(v):,} splats).")
        centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        scales = (np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)) * scale_mult).astype(np.float32)

        # Convert quaternion (rot_0=w, rot_1=x, rot_2=y, rot_3=z) to Rerun xyzw
        quats = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], axis=-1).astype(np.float32)
        norms = np.linalg.norm(quats, axis=-1, keepdims=True)
        quats = np.where(norms > 1e-8, quats / norms, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))

        # Base RGB from spherical harmonics C0
        r = np.clip(0.5 + SH_C0 * v["f_dc_0"], 0.0, 1.0) * 255.0
        g = np.clip(0.5 + SH_C0 * v["f_dc_1"], 0.0, 1.0) * 255.0
        b = np.clip(0.5 + SH_C0 * v["f_dc_2"], 0.0, 1.0) * 255.0

        # Opacity (sigmoid) with opacity multiplier to prevent see-through output
        raw_opacity = 1.0 / (1.0 + np.exp(-v["opacity"]))
        boosted_opacity = np.clip(raw_opacity * opacity_mult, 0.0, 1.0) * 255.0
        colors = np.stack([r, g, b, boosted_opacity], axis=-1).astype(np.uint8)

        rr.log(
            "world/gaussians",
            rr.GaussianSplats3D(
                centers=centers,
                scales=scales,
                quaternions=quats,
                colors=colors,
            ),
            static=True,
        )
    else:
        # Standard point cloud PLY
        print(f"[Rerun] Detected point cloud ({len(v):,} points).")
        pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.uint8)
        else:
            colors = np.full((len(pts), 3), 200, dtype=np.uint8)

        # Set radius so points render as solid 3D spheres instead of 1-pixel see-through dots
        bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) if len(pts) > 0 else 1.0
        radius = max(0.005, float(bbox_diag / 400.0))

        rr.log(
            "world/pointcloud",
            rr.Points3D(
                positions=pts,
                colors=colors,
                radii=radius,
            ),
            static=True,
        )


def visualize(target_path="test_workspace", opacity_mult: float = 3.0, scale_mult: float = 1.0):
    """
    Visualize 3D Gaussian Splats (.ply) and camera poses in Rerun.
    target_path: Path to workspace directory or .ply file.
    opacity_mult: Multiplier for splat opacity (default 3.0 to make scene opaque/solid).
    scale_mult: Multiplier for splat size/scales.
    """
    target = Path(target_path).resolve()

    if target.is_dir():
        workspace = target
        ply_file = workspace / "scene_3dgs.ply"
        if not ply_file.exists():
            ply_file = workspace / "sparse" / "0" / "points3D.ply"
    else:
        ply_file = target
        workspace = target.parent

    if not ply_file.exists():
        raise FileNotFoundError(f"Could not find 3D PLY file at {ply_file}")

    print(f"[Rerun] Initializing viewer for: {ply_file.name}")
    try:
        from rerun import bindings
        bindings.disconnect_orphaned_recordings()
    except Exception:
        pass
    try:
        rr.disconnect()
    except Exception:
        pass

    try:
        rr.init("3dgs_visualizer", spawn=True)
    except RuntimeError:
        # If previous window was closed, force new recording & spawn
        from rerun import bindings
        try:
            bindings.disconnect_orphaned_recordings()
            bindings.new_recording(
                application_id="3dgs_visualizer",
                recording_id=None,
                make_default=True,
                make_thread_default=False,
                default_enabled=True,
                send_properties=True,
                batcher_config=None,
            )
            from rerun.sinks import spawn as _spawn
            _spawn(default_blueprint=None)
        except Exception as e:
            print(f"[Rerun] Fallback init warning: {e}")

    # Set coordinate system convention to RDF (Right, Down, Forward - OpenCV standard)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # 1. Log the 3D model with boosted opacity
    log_ply_file(ply_file, opacity_mult=opacity_mult, scale_mult=scale_mult)

    # 2. Log camera poses, intrinsics, and images if transforms.json is available
    transforms_file = workspace / "transforms.json"
    if transforms_file.exists():
        print(f"[Rerun] Logging camera poses from {transforms_file.name}...")
        with open(transforms_file, "r") as f:
            data = json.load(f)

        for i, frame in enumerate(data.get("frames", [])):
            cam_path = f"world/cameras/camera_{i}"
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            # Camera extrinsic pose (camera-to-world)
            rr.log(
                cam_path,
                rr.Transform3D(
                    translation=c2w[:3, 3],
                    mat3x3=c2w[:3, :3],
                    relation=rr.TransformRelation.ParentFromChild,
                ),
                static=True,
            )

            # Camera intrinsics (pinhole)
            K = np.array([
                [frame["fl_x"], 0, frame["cx"]],
                [0, frame["fl_y"], frame["cy"]],
                [0, 0, 1]
            ], dtype=np.float32)

            rr.log(
                f"{cam_path}/pinhole",
                rr.Pinhole(
                    image_from_camera=K,
                    width=int(frame["w"]),
                    height=int(frame["h"]),
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
                static=True,
            )

            # Log camera RGB image if found
            img_rel = frame.get("file_path", "")
            img_path = workspace / img_rel
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rr.log(f"{cam_path}/pinhole/image", rr.Image(img_rgb), static=True)

    print(f"[Rerun] Scene loaded successfully! Displaying in Rerun viewer.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting Visualizer with Rerun")
    parser.add_argument("target", nargs="?", default="test_workspace",
                        help="Path to workspace folder or .ply file")
    parser.add_argument("--opacity_mult", type=float, default=3.0,
                        help="Multiplier for splat opacity (default 3.0 to reduce transparency)")
    parser.add_argument("--scale_mult", type=float, default=1.0,
                        help="Multiplier for splat scale/size (default 1.0)")
    args = parser.parse_args()

    visualize(args.target, opacity_mult=args.opacity_mult, scale_mult=args.scale_mult)
