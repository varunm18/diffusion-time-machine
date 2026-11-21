"""Skip the 3D-visualization tests when their heavy dependencies are missing.

The viz code (src/viz) imports trimesh, mini_dust3r, rerun and friends, which
are not part of the basic install; they are only needed for the 3D pipeline.
Without this file, one missing package makes pytest crash while collecting,
which hides every other test. Here we skip these files instead.
"""

import importlib.util

# Everything src/viz imports at module level that is not in requirements for the data side.
_NEEDED = ("trimesh", "mini_dust3r", "rerun", "plyfile", "cv2", "scipy")

collect_ignore_glob = []
if any(importlib.util.find_spec(name) is None for name in _NEEDED):
    collect_ignore_glob = ["test_*.py"]
