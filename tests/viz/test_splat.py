"""
Unit tests for src.viz.splat (msplat Metal 3D Gaussian Splatting trainer).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.viz.splat import MSplat


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
