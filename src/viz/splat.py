"""
3D Gaussian Splatting Optimization Backends:
- MSplat: Apple Metal compute acceleration (macOS Apple Silicon)
- GSplat: CUDA rasterization and adaptive density control (Windows / Linux)
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None

try:
    import msplat
except ImportError:
    msplat = None

try:
    import gsplat
except ImportError:
    gsplat = None


def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute 1 - SSIM loss between two (H, W, 3) images normalized to [0, 1].
    """
    if img1.shape[0] < window_size or img1.shape[1] < window_size:
        return torch.tensor(0.0, device=img1.device)
    t1 = img1.permute(2, 0, 1).unsqueeze(0)
    t2 = img2.permute(2, 0, 1).unsqueeze(0)

    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(t1, window, padding=pad, groups=3)
    mu2 = torch.nn.functional.conv2d(t2, window, padding=pad, groups=3)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(t1 * t1, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(t2 * t2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(t1 * t2, window, padding=pad, groups=3) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return 1.0 - ssim_map.mean()


class MSplat:
    """
    3D Gaussian Splatting trainer utilizing Apple Metal compute shaders.
    """

    def __init__(
        self,
        iterations: int = 7000,
        num_downscales: int = 0,
        eval_mode: bool = True,
    ):
        """
        Args:
            iterations: Number of optimization steps (default: 7000).
            num_downscales: Image resolution downscale levels (0 = full resolution).
            eval_mode: Whether to evaluate PSNR/SSIM periodically on test views.
        """
        self.iterations = iterations
        self.num_downscales = num_downscales
        self.eval_mode = eval_mode

    def train(
        self,
        workspace_dir: str | Path,
        output_ply: str | Path | None = None,
        callback_every: int = 100,
    ) -> Path:
        """
        Train 3D Gaussian Splats from a workspace containing transforms.json and points3D.ply.

        Args:
            workspace_dir: Directory containing images, transforms.json, and sparse/0/points3D.ply.
            output_ply: Custom path for exported .ply (default: <workspace>/scene_3dgs.ply).
            callback_every: Print progress every N iterations.

        Returns:
            Path to the exported .ply file.
        """
        if msplat is None:
            raise ImportError(
                "msplat is required for Apple Metal training (macOS Apple Silicon only). "
                "For Windows/Linux CUDA, use GSplat."
            )

        workspace = Path(workspace_dir).resolve()
        output_ply = Path(output_ply or workspace / "scene_3dgs.ply").resolve()

        print(f"[msplat] Loading dataset from {workspace}...")
        dataset = msplat.load_dataset(str(workspace), eval_mode=self.eval_mode)

        print(f"[msplat] Starting Metal-accelerated 3DGS optimization ({self.iterations} iterations)...")
        config = msplat.TrainingConfig(iterations=self.iterations, num_downscales=self.num_downscales)
        trainer = msplat.GaussianTrainer(dataset, config)
        trainer.train(
            lambda s: print(f"step={s.iteration} splats={s.splat_count:,}"),
            callback_every=callback_every,
        )

        # Export both .ply and lightweight .splat formats
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        trainer.export_ply(str(output_ply))
        output_splat = output_ply.with_suffix(".splat")
        trainer.export_splat(str(output_splat))
        print(f"[msplat] Success! 3DGS scene saved to {output_ply} and {output_splat}")
        return output_ply


class GSplat:
    """
    3D Gaussian Splatting trainer utilizing gsplat (for Windows/Linux with CUDA).
    """

    def __init__(
        self,
        iterations: int = 7000,
        lr: float = 1.6e-4,
        max_points: int = 100_000,
        device: str | None = None,
    ):
        """
        Args:
            iterations: Number of optimization steps (default: 7000).
            lr: Base position learning rate for Gaussians (default: 1.6e-4).
            max_points: Maximum initial points sampled from sparse point cloud (default: 100,000).
            device: Compute device ('cuda' or 'cpu').
        """
        self.iterations = iterations
        self.lr = lr
        self.max_points = max_points
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        workspace_dir: str | Path,
        output_ply: str | Path | None = None,
        callback_every: int = 100,
    ) -> Path:
        """
        Train 3D Gaussian Splats from a workspace containing transforms.json and points3D.ply.

        Args:
            workspace_dir: Directory containing images, transforms.json, and sparse/0/points3D.ply.
            output_ply: Custom path for exported .ply (default: <workspace>/scene_3dgs.ply).
            callback_every: Print progress every N iterations.

        Returns:
            Path to the exported .ply file.
        """
        if gsplat is None:
            raise ImportError(
                "gsplat is required for CUDA/Windows training. Install with: pip install gsplat"
            )

        workspace = Path(workspace_dir).resolve()
        output_ply = Path(output_ply or workspace / "scene_3dgs.ply").resolve()
        output_splat = output_ply.with_suffix(".splat")

        print(f"[gsplat] Loading dataset from {workspace}...")
        with open(workspace / "transforms.json", "r") as f:
            meta = json.load(f)

        cameras = []
        for frame in meta.get("frames", []):
            img_path = workspace / frame["file_path"]
            if not img_path.exists():
                continue
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            fl_x = float(frame.get("fl_x", meta.get("fl_x", 800.0)))
            fl_y = float(frame.get("fl_y", meta.get("fl_y", 800.0)))
            cx = float(frame.get("cx", meta.get("cx", 400.0)))
            cy = float(frame.get("cy", meta.get("cy", 300.0)))
            w, h = int(frame.get("w", meta.get("w", 800))), int(frame.get("h", meta.get("h", 600)))

            # Cache image into memory once to eliminate per-step disk / Google Drive I/O latency
            img = Image.open(img_path).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h))
            img_tensor = torch.tensor(np.array(img, dtype=np.float32) / 255.0, device=self.device)

            cameras.append({
                "c2w": c2w,
                "w2c": torch.linalg.inv(c2w).to(self.device),
                "K": torch.tensor([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device),
                "w": w, "h": h, "image": img_tensor,
            })

        if not cameras:
            raise FileNotFoundError(f"No valid camera frames found in {workspace}")

        # Initialize Gaussians from sparse point cloud (or random initialization fallback)
        ply_path = workspace / "sparse" / "0" / "points3D.ply"
        if ply_path.exists():
            v = PlyData.read(str(ply_path))["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
            colors = (
                np.stack([v["red"], v["green"], v["blue"]], axis=-1) / 255.0
                if "red" in v
                else np.full_like(pts, 0.5)
            ).astype(np.float32)
        else:
            pts = (np.random.randn(1000, 3) * 0.5).astype(np.float32)
            colors = np.full_like(pts, 0.5)

        # Subsample initial point cloud if dense (e.g. mini-DUSt3R produces ~1.5M points)
        N_orig = len(pts)
        if N_orig > self.max_points:
            indices = np.random.choice(N_orig, self.max_points, replace=False)
            pts = pts[indices]
            colors = colors[indices]
            print(f"[gsplat] Subsampled initial point cloud from {N_orig:,} to {len(pts):,} points")

        # Estimate scene scale from camera locations or point cloud bounds
        if len(cameras) > 1:
            cam_centers = torch.stack([cam["c2w"][:3, 3] for cam in cameras]).cpu().numpy()
            center = np.mean(cam_centers, axis=0)
            scene_scale = float(np.max(np.linalg.norm(cam_centers - center, axis=1)) * 1.1)
        else:
            scene_scale = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        scene_scale = max(scene_scale, 0.1)

        N = len(pts)
        # Compute nearest-neighbor distances for data-driven initial Gaussian scale
        if N >= 4 and KDTree is not None:
            tree = KDTree(pts)
            dists, _ = tree.query(pts, k=4)
            mean_dist = dists[:, 1:].mean(axis=-1)
            mean_dist = np.clip(mean_dist, 1e-5, scene_scale * 0.1)
            init_scales = np.log(mean_dist[:, None].repeat(3, axis=-1))
        else:
            default_scale = max(scene_scale / max(1, N ** (1 / 3)), 1e-3)
            init_scales = np.full((N, 3), np.log(default_scale), dtype=np.float32)

        pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device)
        scales_t = torch.tensor(init_scales, dtype=torch.float32, device=self.device)
        sh0_val = (torch.tensor(colors, dtype=torch.float32, device=self.device) - 0.5) / 0.28209479177387814

        quats_t = torch.zeros((N, 4), device=self.device)
        quats_t[:, 0] = 1.0

        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts_t),
            "scales": torch.nn.Parameter(scales_t),
            "quats": torch.nn.Parameter(quats_t),
            "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=self.device)),
            "sh0": torch.nn.Parameter(sh0_val.unsqueeze(1)),
        })

        lr_scale = self.lr / 1.6e-4
        optimizers = {
            "means": torch.optim.Adam([params["means"]], lr=self.lr * scene_scale),
            "scales": torch.optim.Adam([params["scales"]], lr=5e-3 * lr_scale),
            "quats": torch.optim.Adam([params["quats"]], lr=1e-3 * lr_scale),
            "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2 * lr_scale),
            "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3 * lr_scale),
        }
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(1, self.iterations))
        )
        strategy = gsplat.DefaultStrategy(
            refine_stop_iter=max(1000, self.iterations - 1000),
            verbose=False,
        )
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)

        print(f"[gsplat] Starting CUDA-accelerated 3DGS optimization ({self.iterations} iterations)...")
        for step in range(self.iterations):
            cam = cameras[torch.randint(0, len(cameras), (1,)).item()]
            gt = cam["image"]

            render_colors, _, info = gsplat.rasterization(
                means=params["means"],
                quats=torch.nn.functional.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["sh0"],
                sh_degree=0,
                viewmats=cam["w2c"].unsqueeze(0),
                Ks=cam["K"].unsqueeze(0),
                width=cam["w"],
                height=cam["h"],
                packed=True,
            )

            pred = render_colors[0]
            l1 = torch.nn.functional.l1_loss(pred, gt)
            ssim = _compute_ssim(pred, gt)
            loss = 0.8 * l1 + 0.2 * ssim

            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            scheduler.step()
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

            with torch.no_grad():
                # Enforce maximum aspect ratio of 10:1 to prevent needle degeneration
                max_log_s = params["scales"].max(dim=-1, keepdim=True).values
                params["scales"].copy_(torch.maximum(params["scales"], max_log_s - 2.302585))

            if callback_every and step % callback_every == 0:
                print(f"step={step} splats={len(params['means']):,} loss={loss.item():.4f}")

        # Export PLY and SPLAT formats
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        shN = torch.zeros((len(params["means"]), 0, 3), device=self.device)
        gsplat.export_splats(
            means=params["means"].detach(),
            scales=params["scales"].detach(),
            quats=torch.nn.functional.normalize(params["quats"].detach(), dim=-1),
            opacities=params["opacities"].detach(),
            sh0=params["sh0"].detach(),
            shN=shN,
            format="ply",
            save_to=str(output_ply),
        )
        gsplat.export_splats(
            means=params["means"].detach(),
            scales=params["scales"].detach(),
            quats=torch.nn.functional.normalize(params["quats"].detach(), dim=-1),
            opacities=params["opacities"].detach(),
            sh0=params["sh0"].detach(),
            shN=shN,
            format="splat",
            save_to=str(output_splat),
        )
        print(f"[gsplat] Success! 3DGS scene saved to {output_ply} and {output_splat}")
        return output_ply


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splats with msplat or gsplat")
    parser.add_argument("--workspace", type=str, default="workspace", help="Path to workspace folder")
    parser.add_argument("--output_ply", type=str, default=None, help="Path for exported .ply file")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "msplat", "gsplat"], help="Backend trainer")
    args = parser.parse_args()

    use_backend = args.backend
    if use_backend == "auto":
        use_backend = "gsplat" if torch.cuda.is_available() else "msplat"

    trainer = GSplat(iterations=args.iterations) if use_backend == "gsplat" else MSplat(iterations=args.iterations)
    trainer.train(args.workspace, args.output_ply)
