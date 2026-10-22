"""
Unit tests for scripts/run_3d_pipeline.py CLI orchestration.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.run_3d_pipeline as pipeline


@patch("scripts.run_3d_pipeline.DUST3R")
@patch("scripts.run_3d_pipeline.MSplat")
@patch("scripts.run_3d_pipeline.visualize")
def test_pipeline_reconstruction_only(mock_viz, mock_msplat, mock_dust3r, tmp_path: Path):
    """Test running the CLI pipeline with only reconstruction enabled."""
    img_dir = tmp_path / "sample_imgs"
    ws = tmp_path / "sample_ws"
    img_dir.mkdir()

    test_args = [
        "run_3d_pipeline.py",
        "--image_dir", str(img_dir),
        "--workspace", str(ws),
        "--device", "cpu",
    ]

    with patch.object(sys, "argv", test_args):
        pipeline.main()

    # Verify DUST3R was instantiated and run
    mock_dust3r.assert_called_once_with(device="cpu", view_reconstruction=False)
    mock_dust3r.return_value.run.assert_called_once_with(
        image_dir=img_dir.resolve(),
        output_workspace=ws.resolve(),
    )

    # Verify MSplat and visualize were NOT called
    mock_msplat.assert_not_called()
    mock_viz.assert_not_called()


@patch("scripts.run_3d_pipeline.DUST3R")
@patch("scripts.run_3d_pipeline.MSplat")
@patch("scripts.run_3d_pipeline.visualize")
def test_pipeline_with_msplat(mock_viz, mock_msplat, mock_dust3r, tmp_path: Path):
    """Test running pipeline with --msplat flag enabled."""
    img_dir = tmp_path / "imgs"
    ws = tmp_path / "ws"
    img_dir.mkdir()

    test_args = [
        "run_3d_pipeline.py",
        "--image_dir", str(img_dir),
        "--workspace", str(ws),
        "--msplat",
    ]

    with patch.object(sys, "argv", test_args):
        pipeline.main()

    mock_dust3r.assert_called_once()
    mock_msplat.assert_called_once()
    mock_msplat.return_value.train.assert_called_once_with(workspace_dir=ws.resolve())
    mock_viz.assert_not_called()


@patch("scripts.run_3d_pipeline.DUST3R")
@patch("scripts.run_3d_pipeline.MSplat")
@patch("scripts.run_3d_pipeline.visualize")
def test_pipeline_full_workflow_with_multipliers(mock_viz, mock_msplat, mock_dust3r, tmp_path: Path):
    """Test full pipeline execution with reconstruction, training, and custom viewer multipliers."""
    img_dir = tmp_path / "imgs"
    ws = tmp_path / "ws"
    img_dir.mkdir()

    test_args = [
        "run_3d_pipeline.py",
        "--image_dir", str(img_dir),
        "--workspace", str(ws),
        "--view_reconstruction",
        "--msplat",
        "--visualize",
        "--opacity_mult", "4.5",
        "--scale_mult", "1.2",
    ]

    with patch.object(sys, "argv", test_args):
        pipeline.main()

    mock_dust3r.assert_called_once_with(device=None, view_reconstruction=True)
    mock_msplat.assert_called_once()
    mock_msplat.return_value.train.assert_called_once_with(workspace_dir=ws.resolve())
    mock_viz.assert_called_once_with(
        ws.resolve(),
        opacity_mult=4.5,
        scale_mult=1.2,
    )

