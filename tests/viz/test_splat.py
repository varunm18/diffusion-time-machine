"""
Unit tests for src.viz.splat (MSplat Metal and GSplat CUDA trainers).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.viz.splat import MSplat, GSplat


def test_msplat_init_defaults():
    """Verify default initialization parameters of MSplat."""
    splatter = MSplat()
    assert splatter.iterations == 7000
    assert splatter.num_downscales == 0
    assert splatter.eval_mode is True


def test_msplat_init_custom():
    """Verify custom parameters of MSplat."""
    splatter = MSplat(iterations=1500, num_downscales=2, eval_mode=False)
    assert splatter.iterations == 1500
    assert splatter.num_downscales == 2
    assert splatter.eval_mode is False


@patch("src.viz.splat.msplat")
def test_msplat_train_default_paths(mock_msplat, tmp_path: Path):
    """Test MSplat training with default output_ply path."""
    mock_dataset = MagicMock()
    mock_trainer = MagicMock()
    mock_msplat.load_dataset.return_value = mock_dataset
    mock_msplat.GaussianTrainer.return_value = mock_trainer

    workspace = tmp_path / "my_workspace"
    workspace.mkdir()

    splatter = MSplat(iterations=1000)
    result_ply = splatter.train(workspace_dir=workspace, callback_every=50)

    expected_ply = workspace / "scene_3dgs.ply"
    expected_splat = workspace / "scene_3dgs.splat"

    assert result_ply == expected_ply
    mock_msplat.load_dataset.assert_called_once_with(str(workspace.resolve()), eval_mode=True)
    mock_msplat.TrainingConfig.assert_called_once_with(iterations=1000, num_downscales=0)
    mock_msplat.GaussianTrainer.assert_called_once_with(mock_dataset, mock_msplat.TrainingConfig.return_value)
    mock_trainer.train.assert_called_once()
    mock_trainer.export_ply.assert_called_once_with(str(expected_ply))
    mock_trainer.export_splat.assert_called_once_with(str(expected_splat))


@patch("src.viz.splat.msplat")
def test_msplat_train_custom_output_ply(mock_msplat, tmp_path: Path):
    """Test MSplat training with an explicitly provided output_ply path."""
    mock_dataset = MagicMock()
    mock_trainer = MagicMock()
    mock_msplat.load_dataset.return_value = mock_dataset
    mock_msplat.GaussianTrainer.return_value = mock_trainer

    workspace = tmp_path / "ws"
    workspace.mkdir()
    custom_ply = tmp_path / "custom_dir" / "model.ply"

    splatter = MSplat(iterations=500, num_downscales=1, eval_mode=False)
    result_ply = splatter.train(workspace_dir=workspace, output_ply=custom_ply)

    expected_splat = custom_ply.with_suffix(".splat")
    assert result_ply == custom_ply.resolve()
    mock_trainer.export_ply.assert_called_once_with(str(custom_ply.resolve()))
    mock_trainer.export_splat.assert_called_once_with(str(expected_splat.resolve()))


def test_gsplat_init_defaults():
    """Verify default initialization parameters of GSplat."""
    splatter = GSplat()
    assert splatter.iterations == 7000
    assert splatter.lr == 1e-3


def test_gsplat_init_custom():
    """Verify custom parameters of GSplat."""
    splatter = GSplat(iterations=2000, lr=5e-4, device="cpu")
    assert splatter.iterations == 2000
    assert splatter.lr == 5e-4
    assert splatter.device == "cpu"


@patch("src.viz.splat.gsplat")
def test_gsplat_train(mock_gsplat, tmp_path: Path):
    """Test GSplat training loop and PLY/SPLAT exports."""
    workspace = tmp_path / "ws_gsplat"
    workspace.mkdir()

    # Create dummy image and transforms.json
    img_dir = workspace / "images"
    img_dir.mkdir()
    dummy_img = img_dir / "frame.jpg"
    from PIL import Image
    Image.new("RGB", (32, 32), color="red").save(dummy_img)

    meta = {
        "camera_model": "OPENCV",
        "frames": [
            {
                "file_path": "images/frame.jpg",
                "fl_x": 50.0, "fl_y": 50.0, "cx": 16.0, "cy": 16.0,
                "w": 32, "h": 32,
                "transform_matrix": np.eye(4).tolist(),
            }
        ]
    }
    (workspace / "transforms.json").write_text(json.dumps(meta))

    mock_strategy = MagicMock()
    mock_strategy.initialize_state.return_value = {}
    mock_gsplat.DefaultStrategy.return_value = mock_strategy

    fake_render = torch.zeros(1, 32, 32, 3, requires_grad=True)
    mock_gsplat.rasterization.return_value = (fake_render, None, {"means2d": fake_render})

    splatter = GSplat(iterations=5, device="cpu")
    out_ply = splatter.train(workspace_dir=workspace, callback_every=2)

    assert out_ply == (workspace / "scene_3dgs.ply").resolve()
    assert mock_gsplat.rasterization.call_count == 5
    assert mock_strategy.step_pre_backward.call_count == 5
    assert mock_strategy.step_post_backward.call_count == 5
    assert mock_gsplat.export_splats.call_count == 2


def test_msplat_missing_raises_error():
    """Verify informative ImportError when msplat is not available."""
    with patch("src.viz.splat.msplat", None), patch.dict("sys.modules", {"msplat": None}):
        with pytest.raises(ImportError, match="msplat is required"):
            MSplat()


def test_gsplat_missing_raises_error():
    """Verify informative ImportError when gsplat is not available."""
    with patch("src.viz.splat.gsplat", None), patch.dict("sys.modules", {"gsplat": None}):
        with pytest.raises(ImportError, match="gsplat is required"):
            GSplat()

