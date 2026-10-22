"""
Unit tests for src.viz.viewer (Rerun 3D Gaussian Splatting and scene visualizer).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.viz.viewer import SH_C0, log_ply_file, visualize, visualize_splats


def test_sh_constant():
    """Verify spherical harmonics degree 0 normalization constant."""
    assert abs(SH_C0 - 0.28209479177387814) < 1e-8


def test_visualize_alias():
    """Verify backward compatibility alias."""
    assert visualize_splats is visualize


def test_visualize_missing_path():
    """Verify visualize raises FileNotFoundError when target path doesn't exist."""
    with pytest.raises(FileNotFoundError):
        visualize("non_existent_path_xyz_123")


def test_visualize_empty_workspace_missing_ply(tmp_path: Path):
    """Verify visualize raises FileNotFoundError when workspace has no PLY file."""
    ws = tmp_path / "empty_ws"
    ws.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not find 3D PLY file"):
        visualize(ws)


@patch("src.viz.viewer.rr")
@patch("src.viz.viewer.PlyData.read")
def test_log_ply_file_gaussian_splats(mock_ply_read, mock_rr, tmp_path: Path):
    """Verify logging 3D Gaussian Splat PLY with rotation, scales, colors, and opacity boost."""
    fake_ply = tmp_path / "scene_3dgs.ply"
    fake_ply.touch()

    # Create mock vertex data with 3DGS fields
    n_points = 5
    names = [
        "x", "y", "z",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "scale_0", "scale_1", "scale_2",
        "opacity",
        "f_dc_0", "f_dc_1", "f_dc_2",
    ]
    mock_vertex = {
        name: np.ones(n_points, dtype=np.float32) for name in names
    }
    # Provide property name simulation
    mock_prop_list = [MagicMock(name=name) for name in names]
    for p, name in zip(mock_prop_list, names):
        p.name = name

    mock_vertex_element = MagicMock()
    mock_vertex_element.properties = mock_prop_list
    mock_vertex_element.__getitem__.side_effect = lambda k: mock_vertex[k]
    mock_vertex_element.__len__.return_value = n_points

    mock_ply = MagicMock()
    mock_ply.__getitem__.return_value = mock_vertex_element
    mock_ply_read.return_value = mock_ply

    log_ply_file(fake_ply, opacity_mult=4.0, scale_mult=1.5)

    assert mock_rr.log.called
    call_args = mock_rr.log.call_args_list[0]
    entity_path, splats_obj = call_args[0][0], call_args[0][1]
    assert entity_path == "world/gaussians"
    assert call_args[1]["static"] is True


@patch("src.viz.viewer.rr")
@patch("src.viz.viewer.PlyData.read")
def test_log_ply_file_pointcloud(mock_ply_read, mock_rr, tmp_path: Path):
    """Verify fallback logging for standard point cloud PLY."""
    fake_ply = tmp_path / "points3D.ply"
    fake_ply.touch()

    # Create mock vertex data without rotation or scale attributes
    n_points = 10
    names = ["x", "y", "z", "red", "green", "blue"]
    mock_vertex = {
        "x": np.zeros(n_points, dtype=np.float32),
        "y": np.ones(n_points, dtype=np.float32),
        "z": np.full(n_points, 2.0, dtype=np.float32),
        "red": np.full(n_points, 255, dtype=np.uint8),
        "green": np.full(n_points, 128, dtype=np.uint8),
        "blue": np.full(n_points, 64, dtype=np.uint8),
    }
    mock_prop_list = [MagicMock(name=name) for name in names]
    for p, name in zip(mock_prop_list, names):
        p.name = name

    mock_vertex_element = MagicMock()
    mock_vertex_element.properties = mock_prop_list
    mock_vertex_element.__getitem__.side_effect = lambda k: mock_vertex[k]
    mock_vertex_element.__len__.return_value = n_points

    mock_ply = MagicMock()
    mock_ply.__getitem__.return_value = mock_vertex_element
    mock_ply_read.return_value = mock_ply

    log_ply_file(fake_ply, scale_mult=1.0)

    assert mock_rr.log.called
    call_args = mock_rr.log.call_args_list[0]
    entity_path = call_args[0][0]
    assert entity_path == "world/pointcloud"
    assert call_args[1]["static"] is True


@patch("src.viz.viewer.rr")
@patch("src.viz.viewer.log_ply_file")
def test_visualize_workspace_with_cameras(mock_log_ply, mock_rr, tmp_path: Path):
    """Verify visualize logs coordinate axes, PLY, and camera frames from transforms.json."""
    ws = tmp_path / "scene_ws"
    ws.mkdir()
    ply = ws / "scene_3dgs.ply"
    ply.touch()

    # Create camera image
    img_dir = ws / "images"
    img_dir.mkdir()
    (img_dir / "frame_0.png").write_bytes(b"dummy_png")

    # Create transforms.json
    transforms_data = {
        "camera_model": "OPENCV",
        "fl_x": 800.0,
        "fl_y": 800.0,
        "cx": 400.0,
        "cy": 300.0,
        "w": 800,
        "h": 600,
        "frames": [
            {
                "file_path": "images/frame_0.png",
                "transform_matrix": np.eye(4).tolist(),
            }
        ],
    }
    (ws / "transforms.json").write_text(json.dumps(transforms_data))

    visualize(ws, opacity_mult=3.0, scale_mult=1.0)

    mock_log_ply.assert_called_once_with(ply, opacity_mult=3.0, scale_mult=1.0)
    assert mock_rr.init.called
    # Check that world coordinates and camera transforms/pinholes were logged
    logged_entities = [call[0][0] for call in mock_rr.log.call_args_list]
    assert "world" in logged_entities
    assert "world/cameras/camera_0" in logged_entities
    assert "world/cameras/camera_0/pinhole" in logged_entities


@patch("src.viz.viewer.rr")
@patch("src.viz.viewer.log_ply_file")
def test_visualize_direct_ply_path(mock_log_ply, mock_rr, tmp_path: Path):
    """Verify visualize accepts a direct path to a .ply file."""
    ply_file = tmp_path / "standalone.ply"
    ply_file.touch()

    visualize(ply_file, opacity_mult=2.5, scale_mult=0.8)

    mock_log_ply.assert_called_once_with(ply_file, opacity_mult=2.5, scale_mult=0.8)
    assert mock_rr.init.called

