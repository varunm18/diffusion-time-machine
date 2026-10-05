"""Read-only access to MegaScenes (https://megascenes.github.io/).

Modules
-------
s3        Key layout of the public bucket and a robust HTTPS client.
index     Global index files: images, scene ids, reconstruction sizes, scene table.
metadata  Per-image Wikimedia Commons metadata (``raw_metadata.json``).
colmap    Minimal readers for COLMAP binary files (cameras, image poses).
poses     Fetching the registered images + poses of a scene.
"""
