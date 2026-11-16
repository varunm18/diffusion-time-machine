"""Skip the 3D-pipeline script test when the visualization dependencies are missing
(same reason and same package list as tests/viz/conftest.py)."""

import importlib.util

_NEEDED = ("trimesh", "mini_dust3r", "rerun", "plyfile", "cv2", "scipy")

collect_ignore_glob = []
if any(importlib.util.find_spec(name) is None for name in _NEEDED):
    collect_ignore_glob = ["test_*.py"]
