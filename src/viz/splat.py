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

# Module-level references loaded lazily inside __init__ of each trainer
msplat = None
gsplat = None


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
        global msplat
        if msplat is None:
            try:
                import msplat as _msplat
                msplat = _msplat
            except ImportError:
                raise ImportError(
                    "msplat is required for Apple Metal training (macOS Apple Silicon only). "
                    "For Windows/Linux CUDA, use GSplat."
                )

        self.msplat = msplat
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
        workspace = Path(workspace_dir).resolve()
        output_ply = Path(output_ply or workspace / "scene_3dgs.ply").resolve()

        print(f"[msplat] Loading dataset from {workspace}...")
        dataset = self.msplat.load_dataset(str(workspace), eval_mode=self.eval_mode)

        print(f"[msplat] Starting Metal-accelerated 3DGS optimization ({self.iterations} iterations)...")
        config = self.msplat.TrainingConfig(iterations=self.iterations, num_downscales=self.num_downscales)
        trainer = self.msplat.GaussianTrainer(dataset, config)
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
        lr: float = 1e-3,
        device: str | None = None,
    ):
        """
        Args:
            iterations: Number of optimization steps (default: 7000).
            lr: Base learning rate for Gaussian parameters.
            device: Compute device ('cuda' or 'cpu').
        """
        global gsplat
        if gsplat is None:
            try:
                import gsplat as _gsplat
                gsplat = _gsplat
            except ImportError:
                raise ImportError(
                    "gsplat is required for CUDA/Windows training. Install with: pip install gsplat"
                )

        self.gsplat = gsplat
        self.iterations = iterations
        self.lr = lr
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
            cameras.append({
                "w2c": torch.linalg.inv(c2w).to(self.device),
                "K": torch.tensor([[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device),
                "w": w, "h": h, "path": img_path,
            })

        if not cameras:
            raise FileNotFoundError(f"No valid camera frames found in {workspace}")

        # Initialize Gaussians from sparse point cloud (or random initialization fallback)
        ply_path = workspace / "sparse" / "0" / "points3D.ply"
        if ply_path.exists():
            v = PlyData.read(str(ply_path))["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], axis=-1)
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=-1) / 255.0 if "red" in v else np.full_like(pts, 0.5)
        else:
            pts = np.random.randn(1000, 3).astype(np.float32) * 0.5
            colors = np.full_like(pts, 0.5)

        N = len(pts)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=self.device)
        sh0_val = (torch.tensor(colors, dtype=torch.float32, device=self.device) - 0.5) / 0.28209479177387814

        params = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(pts_t),
            "scales": torch.nn.Parameter(torch.log(torch.full((N, 3), 0.01, device=self.device))),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(N, 1)),
            "opacities": torch.nn.Parameter(torch.full((N,), -2.0, device=self.device)),
            "sh0": torch.nn.Parameter(sh0_val.unsqueeze(1)),
        })

        optimizers = {
            "means": torch.optim.Adam([params["means"]], lr=self.lr),
            "scales": torch.optim.Adam([params["scales"]], lr=self.lr * 5),
            "quats": torch.optim.Adam([params["quats"]], lr=self.lr),
            "opacities": torch.optim.Adam([params["opacities"]], lr=self.lr * 10),
            "sh0": torch.optim.Adam([params["sh0"]], lr=self.lr * 2.5),
        }
        strategy = self.gsplat.DefaultStrategy(verbose=False)
        strategy_state = strategy.initialize_state()

        print(f"[gsplat] Starting CUDA-accelerated 3DGS optimization ({self.iterations} iterations)...")
        for step in range(self.iterations):
            cam = cameras[torch.randint(0, len(cameras), (1,)).item()]
            img = Image.open(cam["path"]).convert("RGB")
            if img.size != (cam["w"], cam["h"]):
                img = img.resize((cam["w"], cam["h"]))
            gt = torch.tensor(np.array(img, dtype=np.float32) / 255.0, device=self.device)

            render_colors, _, info = self.gsplat.rasterization(
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
            )

            loss = torch.nn.functional.l1_loss(render_colors[0], gt)
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
            loss.backward()
            strategy.step_post_backward(params, optimizers, strategy_state, step, info)

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            if callback_every and step % callback_every == 0:
                print(f"step={step} splats={len(params['means']):,}")

        # Export PLY and SPLAT formats
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        shN = torch.zeros((len(params["means"]), 0, 3), device=self.device)
        self.gsplat.export_splats(
            means=params["means"].detach(),
            scales=params["scales"].detach(),
            quats=torch.nn.functional.normalize(params["quats"].detach(), dim=-1),
            opacities=params["opacities"].detach(),
            sh0=params["sh0"].detach(),
            shN=shN,
            format="ply",
            save_to=str(output_ply),
        )
        self.gsplat.export_splats(
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
