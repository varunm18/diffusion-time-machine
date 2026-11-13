import io

import numpy as np
from PIL import Image, ImageOps

from timemachine.data.cleaning import dhash, duplicate_clusters, hamming_matrix, shape_status


def _photo(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    """A smooth random image (low-resolution noise blown up), standing in for a photo."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8))
    return small.resize(size, Image.Resampling.BICUBIC)


def _bits(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def test_shape_status():
    assert shape_status(1024, 768, 1800, 1350) == "match"       # downscaled, same aspect
    assert shape_status(1024, 775, 1800, 1350) == "match"       # within 1.5%
    assert shape_status(768, 1024, 1800, 1350) == "rotated"
    assert shape_status(1024, 566, 1800, 1350) == "mismatch"    # 1473x814 vs a 4:3 camera
    assert shape_status(1000, 1000, 1200, 1200) == "match"      # square: never "rotated"


def test_dhash_survives_resize_recompression_and_toning():
    photo = _photo(0)
    buf = io.BytesIO()
    photo.resize((320, 240)).save(buf, "JPEG", quality=60)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    sepia = ImageOps.colorize(photo.convert("L"), black="#201000", white="#fff0d0")
    assert _bits(dhash(photo), dhash(recompressed)) <= 4
    assert _bits(dhash(photo), dhash(sepia)) <= 4


def test_dhash_separates_different_photos():
    distances = [_bits(dhash(_photo(0)), dhash(_photo(s))) for s in range(1, 9)]
    assert min(distances) > 12


def test_hamming_matrix_matches_bit_count():
    hashes = [0b0, 0b1011, (1 << 64) - 1]
    d = hamming_matrix(hashes)
    assert d[0, 1] == 3 and d[0, 2] == 64 and d[1, 2] == 61 and (d == d.T).all() and d[1, 1] == 0


def test_duplicate_clusters_follow_chains():
    # 0-1 close, 1-2 close, 0-2 far: one cluster through the chain; 3 alone.
    d = np.array([[0, 3, 9, 30], [3, 0, 4, 31], [9, 4, 0, 29], [30, 31, 29, 0]])
    assert duplicate_clusters(d, max_distance=6).tolist() == [0, 0, 0, 3]
    assert duplicate_clusters(d, max_distance=3).tolist() == [0, 0, 2, 3]
    assert duplicate_clusters(np.zeros((0, 0)), 6).tolist() == []
