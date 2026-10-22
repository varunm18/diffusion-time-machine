"""
Metal-Accelerated 3D Gaussian Splatting Optimization.

Trains a 3D Gaussian Splatting scene on Apple Silicon via `msplat`,
exporting both standard PLY checkpoints and compact binary .splat files.
"""

from pathlib import Path
import msplat


class MSplat:
    """
    3D Gaussian Splatting trainer utilizing Apple Metal compute shaders.
    """

    def __init__(self, iterations: int = 7000, num_downscales: int = 0, eval_mode: bool = True):
        """
        Args:
            iterations: Number of optimization steps (default: 7000).
            num_downscales: Image resolution downscale levels (0 = full resolution).
            eval_mode: Whether to evaluate PSNR/SSIM periodically on test views.
        """
        self.iterations = iterations
        self.num_downscales = num_downscales
        self.eval_mode = eval_mode

    def train(self, workspace_dir: str | Path, output_ply: str | Path | None = None, callback_every: int = 100):
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
        if output_ply is None:
            output_ply = workspace / "scene_3dgs.ply"
        else:
            output_ply = Path(output_ply).resolve()

        print(f"[msplat] Loading dataset from {workspace}...")
        dataset = msplat.load_dataset(str(workspace), eval_mode=self.eval_mode)

        print(f"[msplat] Starting Metal-accelerated 3DGS optimization ({self.iterations} iterations)...")
        config = msplat.TrainingConfig(iterations=self.iterations, num_downscales=self.num_downscales)
        trainer = msplat.GaussianTrainer(dataset, config)

        # Run training loop on Metal GPU
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splats with msplat")
    parser.add_argument("--workspace", type=str, default="workspace",
                        help="Path to workspace folder with transforms.json and sparse/0/points3D.ply")
    parser.add_argument("--output_ply", type=str, default=None,
                        help="Path for exported .ply file (default: <workspace>/scene_3dgs.ply)")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Number of training iterations (default: 7000)")
    args = parser.parse_args()

    splatter = MSplat(iterations=args.iterations)
    splatter.train(args.workspace, args.output_ply)
