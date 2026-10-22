"""
Unit tests for src.viz.reconstruction (mini-DUSt3R 3D reconstruction module).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.viz.reconstruction import DUST3R, save_transforms


def test_save_transforms(tmp_path: Path):
    """Test saving camera frames and intrinsics to transforms.json."""
    output_path = tmp_path / "subfolder" / "transforms.json"
    frames = [
        {
            "file_path": "images/frame_001.jpg",
            "fl_x": 1000.0,
            "fl_y": 1000.0,
            "cx": 500.0,
            "cy": 500.0,
            "w": 1000,
            "h": 1000,
            "transform_matrix": np.eye(4).tolist(),
        }
    ]

    save_transforms(frames, output_path)

    assert output_path.exists()
    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["camera_model"] == "OPENCV"
    assert len(data["frames"]) == 1
    assert data["frames"][0]["file_path"] == "images/frame_001.jpg"
    assert data["frames"][0]["fl_x"] == 1000.0


@patch("src.viz.reconstruction.AsymmetricCroCo3DStereo.from_pretrained")
def test_dust3r_init_custom_device(mock_pretrained):
    """Test DUST3R initialization with explicit device and model."""
    mock_model = MagicMock()
    mock_pretrained.return_value.to.return_value = mock_model

    dust3r = DUST3R(device="cpu", model_name="custom/model_ckpt", view_output=True)

    assert dust3r.device == "cpu"
    assert dust3r.view_output is True
    assert dust3r.model == mock_model
    mock_pretrained.assert_called_once_with("custom/model_ckpt")


@patch("src.viz.reconstruction.AsymmetricCroCo3DStereo.from_pretrained")
def test_dust3r_run_missing_image_dir(mock_pretrained, tmp_path: Path):
    """Test that DUST3R.run raises FileNotFoundError if image_dir does not exist."""
    dust3r = DUST3R(device="cpu")
    non_existent = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        dust3r.run(image_dir=non_existent, output_workspace=tmp_path / "ws")


@patch("src.viz.reconstruction.AsymmetricCroCo3DStereo.from_pretrained")
def test_dust3r_run_empty_image_dir(mock_pretrained, tmp_path: Path):
    """Test that DUST3R.run raises FileNotFoundError if no images are in image_dir."""
    dust3r = DUST3R(device="cpu")
    empty_dir = tmp_path / "empty_images"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No images found"):
        dust3r.run(image_dir=empty_dir, output_workspace=tmp_path / "ws")


@patch("src.viz.reconstruction.inferece_dust3r")
@patch("src.viz.reconstruction.AsymmetricCroCo3DStereo.from_pretrained")
def test_dust3r_run_success(mock_pretrained, mock_inference, tmp_path: Path):
    """Test complete DUST3R reconstruction flow with mocked inference."""
    # Create fake input image files
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "001.jpg").write_bytes(b"dummy_image_data")
    (img_dir / "002.png").write_bytes(b"dummy_image_data")

    # Set up mock inference results
    mock_results = MagicMock()
    mock_pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    mock_results.point_cloud.vertices = mock_pts
    mock_results.point_cloud.colors = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    mock_results.point_cloud.export = MagicMock()

    mock_pose = np.eye(4, dtype=np.float32)
    mock_k = np.array([[500.0, 0.0, 250.0], [0.0, 500.0, 250.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mock_rgb = np.zeros((500, 500, 3), dtype=np.uint8)
    mock_depth = np.ones((500, 500), dtype=np.float32)

    mock_results.world_T_cam_b44 = [mock_pose, mock_pose]
    mock_results.K_b33 = [mock_k, mock_k]
    mock_results.rgb_hw3_list = [mock_rgb, mock_rgb]
    mock_results.depth_hw_list = [mock_depth, mock_depth]

    mock_inference.return_value = mock_results

    workspace = tmp_path / "output_ws"
    dust3r = DUST3R(device="cpu", view_output=False)
    results = dust3r.run(image_dir=img_dir, output_workspace=workspace)

    assert results == mock_results
    # Check that image symlink was created
    assert (workspace / "images").exists()

    # Check point cloud export was called
    mock_results.point_cloud.export.assert_called_once()
    assert (workspace / "sparse" / "0").exists()

    # Check transforms.json was created
    transforms_path = workspace / "transforms.json"
    assert transforms_path.exists()
    with open(transforms_path, "r") as f:
        meta = json.load(f)
    assert len(meta["frames"]) == 2
    assert meta["frames"][0]["file_path"] == "images/001.jpg"
    assert meta["frames"][1]["file_path"] == "images/002.png"
    assert meta["frames"][0]["fl_x"] == 500.0


@patch("rerun.init")
@patch("rerun.log")
@patch("src.viz.reconstruction.inferece_dust3r")
@patch("src.viz.reconstruction.AsymmetricCroCo3DStereo.from_pretrained")
def test_dust3r_run_with_view_output(mock_pretrained, mock_inference, mock_rr_log, mock_rr_init, tmp_path: Path):
    """Test DUST3R reconstruction with view_output=True triggering Rerun preview."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "001.jpg").write_bytes(b"dummy_image_data")

    mock_results = MagicMock()
    mock_pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    mock_results.point_cloud.vertices = mock_pts
    mock_results.point_cloud.colors = np.array([[200, 100, 50]], dtype=np.uint8)
    mock_results.point_cloud.export = MagicMock()

    mock_results.world_T_cam_b44 = [np.eye(4, dtype=np.float32)]
    mock_results.K_b33 = [np.eye(3, dtype=np.float32)]
    mock_results.rgb_hw3_list = [np.zeros((10, 10, 3), dtype=np.uint8)]
    mock_results.depth_hw_list = [np.zeros((10, 10), dtype=np.float32)]

    mock_inference.return_value = mock_results

    workspace = tmp_path / "output_ws_preview"
    dust3r = DUST3R(device="cpu", view_output=True)
    dust3r.run(image_dir=img_dir, output_workspace=workspace)

    assert mock_rr_init.called
    assert mock_rr_log.called
    logged_entities = [call[0][0] for call in mock_rr_log.call_args_list]
    assert "world" in logged_entities
    assert "world/pointcloud" in logged_entities
    assert "world/cameras/camera_0" in logged_entities

