import pytest
from pathlib import Path


def test_viz_imports():
    from src.viz import DUST3R, MSplat, visualize, visualize_splats
    assert DUST3R is not None
    assert MSplat is not None
    assert visualize is not None
    assert visualize_splats is not None


def test_msplat_init():
    from src.viz import MSplat
    splatter = MSplat(iterations=500, num_downscales=1, eval_mode=False)
    assert splatter.iterations == 500
    assert splatter.num_downscales == 1
    assert splatter.eval_mode is False


def test_viewer_missing_file():
    from src.viz import visualize
    with pytest.raises(FileNotFoundError):
        visualize("non_existent_workspace_path_12345")
