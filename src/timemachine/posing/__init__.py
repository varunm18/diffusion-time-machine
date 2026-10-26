"""Placing images that COLMAP could not register (mostly old photos) into a scene.

align     similarity-transform fitting (Umeyama + RANSAC) and pose-error metrics
batches   grouping queries with diverse anchor images
vggt      VGGT-Omega wrapper (used read-only from an existing checkout)
register  the registration procedure and its evaluation
"""
