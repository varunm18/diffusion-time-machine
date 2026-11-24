"""Tests for the dataset build script (``scripts/mvgen/build_dataset.py``), on synthetic inputs.

The script runs in our own environment (it needs the MegaScenes download code), so nothing here
needs MVGenMaster: no ``importorskip``. Nothing touches the network or the real dataset; the one
test of ``main()`` replaces the view table and the download with fakes.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from timemachine.data.cameras import c2w_to_colmap

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    """The script as a module (it is not in a package)."""
    spec = importlib.util.spec_from_file_location("build_dataset", REPO / "scripts/mvgen/build_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module             # dataclasses look their module up while a class is built
    spec.loader.exec_module(module)
    return module


build = _load_script()
FIXES = build.DateFixes(fire_start=2019.2849, after_fire_from=2019.2877, misdated=frozenset({"wrong.jpg"}))
# The same with every rule of the fixes file switched on.
RULES = build.DateFixes(fire_start=2019.2849, after_fire_from=2019.2877, misdated=frozenset({"wrong.jpg"}),
                        montages=frozenset({"both.jpg"}), after_fire_min_confidence=0.95,
                        after_fire_verified=frozenset({"checked.jpg"}), exclude_before=1859.0)


# --- helpers -----------------------------------------------------------------------
def camera(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Camera-to-world matrix (OpenCV axes) in a world whose up is +z.

    The camera looks along the horizontal direction ``yaw`` (degrees from +x), tilted up by ``pitch``.
    ``roll`` turns it about its viewing axis so that the world's up appears ``roll`` degrees away from
    the image's up, towards the image's right.
    """
    yaw, pitch, roll = np.radians([yaw, pitch, roll])
    ahead, up = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([0.0, 0.0, 1.0])
    forward = np.cos(pitch) * ahead + np.sin(pitch) * up
    down = np.sin(pitch) * ahead - np.cos(pitch) * up
    right = np.cross(down, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = np.cos(roll) * right - np.sin(roll) * down
    c2w[:3, 1] = np.sin(roll) * right + np.cos(roll) * down
    c2w[:3, 2] = forward
    c2w[:3, 3] = centre
    return c2w


def rotation(axis, degrees: float) -> np.ndarray:
    """Rotation matrix about ``axis`` (Rodrigues' formula)."""
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def angle_difference(a, b) -> np.ndarray:
    """Difference of two angles in degrees, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def picture(seed: int, size: tuple[int, int] = (96, 64)) -> Image.Image:
    """A random grey picture made of large patches, which a 32 x 32 thumbnail keeps."""
    coarse = np.random.default_rng(seed).integers(0, 256, (8, 12), dtype=np.uint8)
    return Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC)


def thumbnail(kind: int, dim: int = 16) -> np.ndarray:
    """A unit, mean-free vector per ``kind``: equal kinds correlate 1, different kinds -1 / (dim - 1)."""
    vector = -np.ones(dim) / dim
    vector[kind] += 1.0
    return vector / np.linalg.norm(vector)


def shared_counts(point_sets: list[set]) -> np.ndarray:
    """Shared point counts of views given the set of points each one sees (the slow, obvious way)."""
    return np.array([[len(a & b) for b in point_sets] for a in point_sets], dtype=np.uint16)


def on_circle(place: int, n_places: int = 16, radius: float = 10.0) -> tuple[float, np.ndarray]:
    """Yaw (degrees, towards the centre) and position of ``place`` on a circle around the origin."""
    angle = 360.0 * place / n_places
    return angle + 180.0, radius * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0])


def duplicate_inputs(rows: list[tuple]) -> tuple:
    """Arguments of ``duplicate_pairs`` for views given as (place, turn, thumbnail kind, points, date_lo, date_hi).

    Views at the same ``place`` (of 16 on a circle of radius 10, the scene's spread) stand 0.01 apart
    and look at the centre, turned by ``turn`` degrees.
    """
    c2w = []
    for index, (place, turn, *_) in enumerate(rows):
        yaw, centre = on_circle(place)
        c2w.append(camera(yaw + turn, centre=centre + [0.0, 0.0, 0.01 * index]))
    thumbs = np.stack([thumbnail(row[2]) for row in rows])
    return (thumbs, np.stack(c2w), shared_counts([row[3] for row in rows]),
            np.array([row[4] for row in rows]), np.array([row[5] for row in rows]))


def write_images_bin(path: Path, images: list[tuple[str, list[int]]], poses: list[tuple] | None = None) -> None:
    """A COLMAP ``images.bin`` with the given (name, 3D point id of each keypoint; -1 for none).

    ``poses`` gives each image's COLMAP (quaternion, translation); by default all images share one pose.
    """
    poses = poses or [((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))] * len(images)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image_id, ((name, point_ids), (qvec, tvec)) in enumerate(zip(images, poses, strict=True), start=1):
            stream.write(struct.pack("<I4d3dI", image_id, *qvec, *tvec, 7))
            stream.write(name.encode("utf-8") + b"\x00")
            stream.write(struct.pack("<Q", len(point_ids)))
            for k, point_id in enumerate(point_ids):
                stream.write(struct.pack("<ddq", k + 0.5, 2 * k + 0.5, point_id))


def write_points3d_bin(path: Path, points: list[tuple[int, tuple[float, float, float], int]]) -> None:
    """A COLMAP ``points3D.bin`` with the given (id, position, track length)."""
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, xyz, track_length in points:
            stream.write(struct.pack("<Q3d3BdQ", point_id, *xyz, 10, 20, 30, 0.5, track_length))
            stream.write(struct.pack(f"<{2 * track_length}I", *range(2 * track_length)))


# --- date fixes and the choice of views -----------------------------------------------
def test_date_fixes_are_validated_and_read_from_json(tmp_path):
    with pytest.raises(ValueError, match="fire_start"):
        build.DateFixes(fire_start=2019.29, after_fire_from=2019.29)
    path = tmp_path / "fixes.json"
    path.write_text(json.dumps({"scene_id": 25512, "fire_start": 2019.2849, "after_fire_from": 2019.2877,
                                "misdated": {"Cathédrale 1.jpg": "shows the spire"}, "source": "a note"}),
                    encoding="utf-8")
    fixes = build.load_date_fixes(path)
    assert fixes == build.DateFixes(2019.2849, 2019.2877, frozenset({"Cathédrale 1.jpg"}))
    assert fixes.after_fire_min_confidence is None and fixes.exclude_before is None     # rules a file may leave out
    with pytest.raises(ValueError, match="scene 25512"):
        build.load_date_fixes(path, scene_id=1)


def test_date_fixes_read_every_rule(tmp_path):
    path = tmp_path / "fixes.json"
    path.write_text(json.dumps({
        "scene_id": 25512, "fire_start": 2019.2849, "after_fire_from": 2019.2877, "misdated": {"wrong.jpg": "why"},
        "montages": {"both.jpg": "before and after"}, "after_fire_min_confidence": 0.95,
        "after_fire_verified": ["checked.jpg"], "after_fire_note": "notes are for the reader",
        "exclude_before": 1859, "duplicates": [["print.jpg", "scan.jpg"], ["a.jpg", "b.jpg"]]}), encoding="utf-8")
    fixes = build.load_date_fixes(path)
    assert fixes == build.DateFixes(
        2019.2849, 2019.2877, misdated=frozenset({"wrong.jpg"}), montages=frozenset({"both.jpg"}),
        after_fire_min_confidence=0.95, after_fire_verified=frozenset({"checked.jpg"}), exclude_before=1859.0,
        duplicates=(("print.jpg", "scan.jpg"), ("a.jpg", "b.jpg")))
    assert fixes.listed_keys == {"wrong.jpg", "both.jpg", "checked.jpg", "print.jpg", "scan.jpg", "a.jpg", "b.jpg"}
    for bad in (dict(after_fire_min_confidence=0.0), dict(after_fire_min_confidence=1.5),
                dict(duplicates=(("a.jpg", "a.jpg"),)), dict(duplicates=(("a.jpg", "b.jpg", "c.jpg"),))):
        with pytest.raises(ValueError):
            build.DateFixes(2019.2849, 2019.2877, **bad)


def test_the_repo_date_fixes_file_loads():
    """The file the build reads by default: the period boundary of spec section 8.2 and the review's date fixes."""
    fixes = build.load_date_fixes(build.DATE_FIXES)
    assert (fixes.fire_start, fixes.after_fire_from) == (2019.2849, 2019.2877)
    assert len(fixes.misdated) == 14 and {"Catedral_de_Notre_Dam.jpg", "Garden_Notre_Dame.jpg",
                                          "20190506_Paris_nd.jpg"} <= fixes.misdated
    assert fixes.montages == {"Notre_Dame-before-the-fire-and-after-the-fire-2019.jpg",
                              "Before_and_after_the_fire_Notre_Dame.jpg"}
    assert fixes.after_fire_min_confidence == 0.95 and len(fixes.after_fire_verified) == 6
    assert fixes.exclude_before == 1859.0
    assert len(fixes.duplicates) == 5 and ("Edouard-Denis_Baldus_Notre_Dame.jpg",
                                           "Édouard_Baldus,_Notre_Dame_de_Paris_-_Getty_Museum.jpg") in fixes.duplicates
    lists = [fixes.misdated, fixes.montages, fixes.after_fire_verified]
    assert sum(len(keys) for keys in lists) == len(frozenset().union(*lists))       # no photo is in two lists


def test_select_views_keeps_old_views_views_from_the_event_on_and_the_base_table():
    dates = [1850.0, 1969.99, 1970.0, 2000.0, 2019.28, 2019.2849, 2019.2877, 2021.0]
    full = pd.DataFrame({"file_key": [f"v{i}" for i in range(len(dates))], "date_mid": dates}, index=range(10, 18))
    kept = build.select_views(full, ["v3", "not in the table"], old_before=1970.0, event_start=2019.2849)
    # 1970.0 is not old, 2019.28 is before the fire: both are kept only if the base table has them (v3 is).
    assert kept["file_key"].tolist() == ["v0", "v1", "v3", "v5", "v6", "v7"]
    assert kept.index.tolist() == list(range(6))


def test_exclusion_reasons_take_the_first_that_applies():
    views = pd.DataFrame([
        # file_key, date_mid, date_confidence, rolled, duplicate_of, k1, k2      -> expected reason
        ("fine.jpg", 2015.0, 0.95, False, -1, 0.1, 0.0),            # ""
        ("wrong.jpg", 2019.2860, 0.5, True, -1, np.nan, 0.0),       # misdated comes before everything else
        ("both.jpg", 2019.2860, 0.5, True, 0, 0.0, 0.0),            # montage beats fire_day
        ("fire.jpg", 2019.2849, 0.5, True, 0, 0.0, 0.0),            # fire_day: the first day counts
        ("after.jpg", 2019.2877, 0.95, False, -1, 0.0, 0.0),        # "": the day after the fire, a confident date
        ("upload.jpg", 2020.5, 0.85, True, 0, 0.0, 0.0),            # uncertain_date beats rolled
        ("checked.jpg", 2020.5, 0.85, False, -1, 0.0, 0.0),         # "": as uncertain, but verified by eye
        ("undated.jpg", 2020.5, np.nan, False, -1, 0.0, 0.0),       # uncertain_date: no confidence at all
        ("modern.jpg", 2015.0, 0.3, False, -1, 0.0, 0.0),           # "": the rule is for after-fire dates only
        ("early.jpg", 1858.9, 0.95, True, 0, 0.0, 0.0),             # before_spire beats rolled
        ("spire.jpg", 1859.0, 0.95, False, -1, 0.0, 0.0),           # "": the year the spire was built is kept
        ("side.jpg", 2015.0, 0.95, True, 0, np.nan, 0.0),           # rolled beats duplicate
        ("copy.jpg", 2015.0, 0.95, False, 0, np.inf, 0.0),          # duplicate beats bad_distortion
        ("lens.jpg", 2015.0, 0.95, False, -1, 0.0, np.nan),         # bad_distortion: k2 is not finite
        ("before.jpg", 2019.2848, 0.95, False, -1, 0.0, 0.0),       # "": just before the fire
    ], columns=["file_key", "date_mid", "date_confidence", "rolled", "duplicate_of", "k1", "k2"])
    expected = ["", "misdated", "montage", "fire_day", "", "uncertain_date", "", "uncertain_date", "", "before_spire",
                "", "rolled", "duplicate", "bad_distortion", ""]
    assert build.exclusion_reasons(views, RULES).tolist() == expected
    assert build.EXCLUSIONS == ("misdated", "montage", "fire_day", "uncertain_date", "before_spire", "rolled",
                                "duplicate", "bad_distortion")
    # Fixes without the optional rules: only the fire's dates and the misdated list apply, and the
    # confidence column is not needed.
    plain = build.exclusion_reasons(views.drop(columns="date_confidence"), FIXES).tolist()
    assert plain == ["", "misdated", "fire_day", "fire_day", "", "rolled", "", "", "", "rolled", "", "rolled",
                     "duplicate", "bad_distortion", ""]


def test_copies_of_a_misdated_photo_are_misdated():
    views = pd.DataFrame([
        # file_key, duplicate_of
        ("first.jpg", -1),                  # 0: the first copy of a photo that is listed under another name
        ("second.jpg", 0),
        ("wrong.jpg", 0),                   # 2: the listed copy
        ("alone.jpg", -1),
        ("other.jpg", -1),                  # 4, 5: another photo, twice
        ("other copy.jpg", 4),
    ], columns=["file_key", "duplicate_of"]).assign(date_mid=1900.0, rolled=False, k1=0.0, k2=0.0)
    assert build.exclusion_reasons(views, FIXES).tolist() == ["misdated", "misdated", "misdated", "", "", "duplicate"]
    listed_first = build.DateFixes(2019.2849, 2019.2877, misdated=frozenset({"other.jpg"}))
    assert build.exclusion_reasons(views, listed_first).tolist() == ["", "duplicate", "duplicate", "", "misdated",
                                                                    "misdated"]
    # A montage is one file: the photos it was made from are not montages.
    montage = build.DateFixes(2019.2849, 2019.2877, montages=frozenset({"wrong.jpg"}))
    assert build.exclusion_reasons(views, montage).tolist() == ["", "duplicate", "montage", "", "", "duplicate"]


# --- sideways views -------------------------------------------------------------------
def test_roll_is_the_rotation_about_the_viewing_axis():
    rolls = [0.0] * 12 + [30.0, -30.0, 90.0, -90.0, 180.0, 60.0, -135.0]
    yaws = np.linspace(0.0, 360.0, len(rolls), endpoint=False)
    c2w = np.stack([camera(yaw, 0.0, roll) for yaw, roll in zip(yaws, rolls)])
    np.testing.assert_allclose(angle_difference(build.roll_angles(c2w), rolls), 0.0, atol=1e-6)


def test_roll_sign_follows_the_scene_up_in_the_image():
    """Positive when the scene's up points to the image's right (the docstring's convention)."""
    c2w = np.stack([camera(yaw) for yaw in (0.0, 90.0, 180.0, 270.0)] + [np.eye(4)])
    c2w[4, :3, :3] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])   # x = world up, looking along +y
    assert build.roll_angles(c2w)[4] == pytest.approx(90.0)


def test_roll_ignores_where_the_camera_looks():
    """Looking up or down does not count as roll, and does not hide it."""
    upright = [camera(yaw, pitch) for yaw in np.linspace(0.0, 360.0, 16, endpoint=False) for pitch in (-20.0, 35.0)]
    c2w = np.stack(upright + [camera(10.0, 25.0, 90.0), camera(200.0, -10.0, -40.0)])
    np.testing.assert_allclose(build.roll_angles(c2w), [0.0] * 32 + [90.0, -40.0], atol=1e-6)


def test_roll_does_not_depend_on_the_world_frame():
    rolls = [0.0] * 9 + [50.0, -90.0, 100.0]
    c2w = np.stack([camera(37.0 * i, 0.0, roll, centre=(i, -i, 2.0)) for i, roll in enumerate(rolls)])
    turned = c2w.copy()
    turned[:, :3, :] = rotation((1.0, 2.0, 3.0), 77.0) @ c2w[:, :3, :]
    np.testing.assert_allclose(build.roll_angles(turned), rolls, atol=1e-6)


# --- duplicate photos -----------------------------------------------------------------
def test_thumbnails_are_unit_vectors_whose_dot_product_is_the_correlation():
    original = picture(0)
    thumb = build.grey_thumbnail(original)
    assert thumb.shape == (build.THUMB_SIZE ** 2,)
    assert abs(thumb.mean()) < 1e-12 and np.linalg.norm(thumb) == pytest.approx(1.0)
    darker = Image.eval(original, lambda value: value // 2 + 60)        # another exposure of the same picture
    assert thumb @ build.grey_thumbnail(darker) > 0.99
    assert thumb @ build.grey_thumbnail(original.resize((480, 320))) > 0.98
    assert thumb @ build.grey_thumbnail(original.convert("RGB")) > 0.999
    assert abs(thumb @ build.grey_thumbnail(picture(1))) < 0.5
    assert not build.grey_thumbnail(Image.new("L", (40, 30), 128)).any()   # nothing to correlate with


def test_load_thumbnails_reads_the_files_in_order(tmp_path):
    paths = []
    for seed in (3, 4, 5):
        paths.append(tmp_path / f"{seed}.png")
        picture(seed).save(paths[-1])
    thumbs = build.load_thumbnails(paths)
    assert thumbs.shape == (3, build.THUMB_SIZE ** 2)
    np.testing.assert_allclose(thumbs[1], build.grey_thumbnail(picture(4)))


def test_pose_differences_are_relative_to_the_scene_spread():
    centres = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (-2, 0, 0), (0, -2, 0)]       # median distance from the middle: 2
    c2w = np.stack([camera(centre=centre) for centre in centres])
    c2w[2, :3, :3] = rotation((1.0, 1.0, 0.0), 30.0) @ c2w[2, :3, :3]
    c2w[4] = camera(roll=90.0, centre=centres[4])                              # roll is a rotation too
    distance, angle = build.pose_differences(c2w)
    assert distance[1, 3] == pytest.approx(2.0) and distance[0, 1] == pytest.approx(1.0)
    assert distance[1, 2] == pytest.approx(np.sqrt(2.0))
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(np.diag(angle), 0.0, atol=1e-5)
    assert angle[0, 2] == pytest.approx(30.0) and angle[0, 4] == pytest.approx(90.0)
    assert angle[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_two_of_three_signs_make_a_duplicate():
    a, b = set(range(100)), set(range(20, 120))             # 80 shared of 120 seen by either: overlap 0.67
    rows = [
        # place, turn, thumbnail, points, date_lo, date_hi
        (0, 0.0, 0, set(range(1000, 1100)), 2010.0, 2010.1),   # 0, 1: same camera and thumbnail      -> duplicate
        (0, 0.0, 0, set(range(1100, 1200)), 2010.0, 2010.1),
        (1, 0.0, 1, a, 2010.0, 2010.1),                        # 2, 3: same camera and points         -> duplicate
        (1, 0.0, 2, b, 2010.0, 2010.1),
        (2, 0.0, 3, {p + 200 for p in a}, 1880.0, 1890.0),     # 4, 5: same thumbnail and points,
        (3, 0.0, 3, {p + 200 for p in b}, 1885.0, 1895.0),     #       cameras apart                  -> duplicate
        (4, 0.0, 4, set(range(2000, 2100)), 2010.0, 2010.1),   # 6, 7: same thumbnail only
        (5, 0.0, 4, set(range(2100, 2200)), 2010.0, 2010.1),
        (6, 0.0, 5, set(range(3000, 3100)), 2010.0, 2010.1),   # 8, 9: same camera only
        (6, 0.0, 6, set(range(3100, 3200)), 2010.0, 2010.1),
        (7, 0.0, 7, {p + 400 for p in a}, 2010.0, 2010.1),     # 10, 11: same points only
        (8, 0.0, 8, {p + 400 for p in b}, 2010.0, 2010.1),
        (9, 0.0, 9, {p + 600 for p in a}, 2010.0, 2010.1),     # 12, 13: all three signs, but other dates
        (9, 0.0, 9, {p + 600 for p in b}, 2015.0, 2015.1),
        (10, 0.0, 10, {p + 800 for p in a}, 1900.0, 1901.0),   # 14, 15: all three, but the date intervals only
        (10, 0.0, 10, {p + 800 for p in b}, 1901.0, 1902.0),   #         touch (they are half-open)
        (11, 0.0, 11, set(range(5000, 5300)), 2010.0, 2010.1),  # 16, 17: a zoom from the same spot: all 90 points
        (11, 0.0, 12, set(range(5000, 5090)), 2010.0, 2010.1),  #         of one are in the other, overlap 0.3
        (12, 0.0, 13, set(range(6000, 6100)), 2010.0, 2010.1),  # 18, 19: same spot and thumbnail, but turned
        (12, 10.0, 13, set(range(6100, 6200)), 2010.0, 2010.1),  #        by 10 degrees: not the same camera
        (13, 0.0, 14, set(), 2010.0, 2010.1),                   # 20, 21: same camera, no points at all
        (13, 0.0, 15, set(), 2010.0, 2010.1),
    ]
    with np.errstate(all="raise"):
        pairs = build.duplicate_pairs(*duplicate_inputs(rows))
    assert pairs.tolist() == [[0, 1], [2, 3], [4, 5]]


def test_duplicate_thresholds_are_arguments():
    # Two views at one spot, turned by 3 degrees, sharing 50 of the 150 points either sees (overlap 0.33);
    # three views elsewhere, from another century, give the scene its size.
    rows = [(0, 0.0, 0, set(range(100)), 2010.0, 2010.1), (0, 3.0, 1, set(range(50, 150)), 2010.0, 2010.1)]
    rows += [(place, 0.0, place, set(range(100 * place, 100 * place + 50)), 1800.0, 1801.0) for place in (4, 8, 12)]
    inputs = duplicate_inputs(rows)
    assert build.duplicate_pairs(*inputs).tolist() == []
    assert build.duplicate_pairs(*inputs, min_point_overlap=0.3).tolist() == [[0, 1]]
    assert build.duplicate_pairs(*inputs, min_point_overlap=0.3, max_rotation_deg=2.0).tolist() == []
    assert build.duplicate_pairs(*inputs, min_corr=-0.5).tolist() == [[0, 1]]      # the thumbnails correlate -1/15
    assert build.duplicate_pairs(*inputs, min_corr=-0.5, max_centre_dist=0.0005).tolist() == []


def test_duplicate_groups_chain_pairs_to_their_first_copy():
    expected = [-1, -1, -1, 2, 1, -1, 1]
    assert build.duplicate_groups(7, np.array([[1, 4], [4, 6], [2, 3]])).tolist() == expected
    assert build.duplicate_groups(7, np.array([[6, 4], [3, 2], [4, 1]])).tolist() == expected   # any order
    # The smallest row joins last: the whole chain still points to it.
    assert build.duplicate_groups(7, np.array([[5, 6], [3, 5], [0, 3]])).tolist() == [-1, -1, -1, 0, -1, 0, 0]
    assert build.duplicate_groups(3, np.empty((0, 2), dtype=int)).tolist() == [-1, -1, -1]
    assert build.duplicate_groups(0, np.empty((0, 2), dtype=int)).tolist() == []


def synthetic_table() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Nine views with one reason to exclude each, their thumbnails and shared point counts."""
    rows = [
        # file_key, place, roll, thumbnail, date_mid, k1
        ("first.jpg", 0, 0.0, 0, 2015.5, 0.0),
        ("copy.jpg", 0, 0.0, 0, 2015.5, 0.0),               # same camera, thumbnail and date as the first
        ("side.jpg", 1, 90.0, 1, 2015.5, 0.0),
        ("wrong.jpg", 2, -90.0, 2, 2020.5, 0.0),            # misdated, and sideways too
        ("fire.jpg", 3, 0.0, 3, 2019.2860, 0.0),
        ("lens.jpg", 4, 0.0, 4, 2015.5, np.nan),
        ("other.jpg", 5, 0.0, 5, 1890.0, -0.2),
        ("tilted.jpg", 6, 44.0, 6, 2015.5, 0.0),            # just under 45 degrees: still upright
        ("more tilted.jpg", 7, -46.0, 7, 2015.5, 0.0),
    ]
    records = []
    for index, (file_key, place, roll, _, date, k1) in enumerate(rows):
        yaw, centre = on_circle(place)
        qvec, tvec = c2w_to_colmap(camera(yaw, 0.0, roll, centre=centre + [0.0, 0.0, 0.01 * index]))
        records.append({"file_key": file_key, "date_lo": date - 0.001, "date_hi": date + 0.001, "date_mid": date,
                        "k1": k1, "k2": 0.0, **dict(zip(("qw", "qx", "qy", "qz"), qvec)),
                        **dict(zip(("tx", "ty", "tz"), tvec))})
    thumbs = np.stack([thumbnail(row[3]) for row in rows])
    shared = shared_counts([set(range(100 * i, 100 * i + 50)) for i in range(len(rows))])
    return pd.DataFrame(records), thumbs, shared


def test_flag_views_adds_the_four_columns():
    views, thumbs, shared = synthetic_table()
    flagged = build.flag_views(views, thumbs, shared, FIXES)
    assert list(flagged.columns) == list(views.columns) + ["rolled", "duplicate_of", "excluded", "usable"]
    assert "usable" not in views                             # the input is left alone
    assert flagged["rolled"].tolist() == [False, False, True, True, False, False, False, False, True]
    assert flagged["duplicate_of"].tolist() == [-1, 0, -1, -1, -1, -1, -1, -1, -1]
    assert flagged["excluded"].tolist() == ["", "duplicate", "rolled", "misdated", "fire_day", "bad_distortion",
                                            "", "", "rolled"]
    assert flagged["usable"].tolist() == [True, False, False, False, False, False, True, True, False]
    assert flagged["rolled"].dtype == bool and flagged["usable"].dtype == bool
    assert pd.api.types.is_integer_dtype(flagged["duplicate_of"])


def test_listed_pairs_are_looked_up_by_file_key():
    keys = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    pairs = build.listed_pairs(keys, [("c.jpg", "a.jpg"), ("b.jpg", "not in the table.jpg"), ("d.jpg", "b.jpg")])
    assert pairs.tolist() == [[2, 0], [3, 1]] and pairs.dtype.kind == "i"
    assert build.listed_pairs(keys, []).shape == (0, 2)


def test_flag_views_chains_listed_duplicates_with_the_ones_it_finds():
    views, thumbs, shared = synthetic_table()
    by_hand = build.DateFixes(2019.2849, 2019.2877, misdated=frozenset({"wrong.jpg"}), duplicates=(
        ("tilted.jpg", "other.jpg"),                # rows 7 and 6: the later row is the copy, whatever the order
        ("other.jpg", "copy.jpg"),                  # row 1 is itself a copy of row 0: one group of four
        ("lens.jpg", "wrong.jpg"),                  # rows 5 and 3: a copy of a misdated photo
        ("first.jpg", "not in the table.jpg")))     # ignored
    flagged = build.flag_views(views, thumbs, shared, by_hand)
    assert flagged["duplicate_of"].tolist() == [-1, 0, -1, -1, -1, 3, 0, 0, -1]
    assert flagged["excluded"].tolist() == ["", "duplicate", "rolled", "misdated", "fire_day", "misdated", "duplicate",
                                            "duplicate", "rolled"]
    assert flagged["usable"].tolist() == [True] + [False] * 8


# --- COLMAP files ---------------------------------------------------------------------
IMAGES = [
    ("commons/Façade ouest/0/pictures/Cathédrale 1.jpg", [5, -1, 9, 5, 12]),
    ("commons/A/0/pictures/skipped.jpg", [1, 2, 3]),
    ("commons/A/0/pictures/no keypoints.jpg", []),
    ("commons/A/0/pictures/nothing triangulated.jpg", [-1, -1]),
    ("commons/B/0/pictures/last.jpg", [12, 3, -1]),
]
POINTS = [(12, (1.0, 2.0, 3.0), 2), (3, (-1.0, 0.5, 0.0), 0), (9, (0.0, 0.0, 7.5), 5), (5, (4.0, 4.0, 4.0), 1),
          (1, (9.0, 8.0, 7.0), 3), (2, (0.1, 0.2, 0.3), 1)]


@pytest.fixture
def colmap_dir(tmp_path) -> Path:
    """A folder with a small COLMAP model: ``IMAGES`` and ``POINTS``."""
    folder = tmp_path / "colmap"
    folder.mkdir()
    write_images_bin(folder / "images.bin", IMAGES)
    write_points3d_bin(folder / "points3D.bin", POINTS)
    return folder


def test_read_observed_points_keeps_triangulated_keypoints_of_the_named_images(colmap_dir):
    wanted = [IMAGES[0][0], IMAGES[2][0], IMAGES[3][0], IMAGES[4][0], "not in the model.jpg"]
    observed = build.read_observed_points(colmap_dir / "images.bin", wanted)
    assert set(observed) == set(wanted[:4])                  # the skipped image and the unknown name are absent
    assert observed[IMAGES[0][0]].tolist() == [5, 9, 5, 12]  # file order, -1 dropped, repeats kept
    assert observed[IMAGES[2][0]].tolist() == [] and observed[IMAGES[3][0]].tolist() == []
    assert observed[IMAGES[4][0]].tolist() == [12, 3]
    assert all(ids.dtype == np.int64 for ids in observed.values())


@pytest.mark.parametrize("wanted", [[IMAGES[4][0]], [IMAGES[0][0]]], ids=["last image read", "last image skipped"])
def test_read_observed_points_refuses_a_damaged_file(colmap_dir, wanted):
    path = colmap_dir / "images.bin"
    whole = path.read_bytes()
    path.write_bytes(whole[:-10])
    with pytest.raises(ValueError, match="end of"):
        build.read_observed_points(path, wanted)
    path.write_bytes(whole + b"\x00\x00")
    with pytest.raises(ValueError, match="not at the end of the file"):
        build.read_observed_points(path, wanted)


def test_read_camera_centres_of_every_registered_image(tmp_path):
    cameras = [camera(30.0 * i, 5.0 * i, 10.0 * i, centre=(i, -2.0 * i, 0.5 * i * i)) for i in range(6)]
    path = tmp_path / "images.bin"
    write_images_bin(path, [(f"image {i}.jpg", [3, -1] * i) for i in range(6)], [c2w_to_colmap(c2w) for c2w in cameras])
    centres = build.read_camera_centres(path)
    assert centres.shape == (6, 3) and centres.dtype == np.float64
    np.testing.assert_allclose(centres, [c2w[:3, 3] for c2w in cameras], atol=1e-12)
    box = build.camera_box(centres, 25512, 0)
    assert json.loads(json.dumps(box)) == box                # plain numbers, ready for scene.json
    assert (box["scene_id"], box["model"], box["n_registered_cameras"]) == (25512, 0, 6)
    np.testing.assert_allclose(box["camera_min"], [0.0, -10.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(box["camera_max"], [5.0, 0.0, 12.5], atol=1e-12)
    assert box["camera_extent"] == pytest.approx(12.5)       # the longest of the three sides
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="end of"):
        build.read_camera_centres(path)
    write_images_bin(path, [])
    assert build.read_camera_centres(path).shape == (0, 3)


def test_load_camera_box_reuses_the_file_of_the_same_scene(colmap_dir, tmp_path):
    cache, nowhere = tmp_path / "scene.json", tmp_path / "no colmap here"
    box = build.load_camera_box(colmap_dir, cache, 25512, 0)
    assert box["n_registered_cameras"] == len(IMAGES) and not cache.exists()    # writing is the caller's business
    with pytest.raises(FileNotFoundError):
        build.load_camera_box(nowhere, cache, 25512, 0)
    cache.write_text(json.dumps({**box, "camera_extent": 7.0}))
    assert build.load_camera_box(nowhere, cache, 25512, 0)["camera_extent"] == 7.0      # read, not parsed
    assert build.load_camera_box(colmap_dir, cache, 25512, 1) == {**box, "model": 1}    # another model: parsed
    assert build.load_camera_box(colmap_dir, cache, 1, 0) == {**box, "scene_id": 1}


def test_read_points3d_sorts_by_id(colmap_dir):
    ids, xyz = build.read_points3d(colmap_dir / "points3D.bin")
    assert ids.tolist() == [1, 2, 3, 5, 9, 12] and ids.dtype == np.int64
    assert xyz.dtype == np.float64 and xyz.shape == (6, 3)
    np.testing.assert_array_equal(xyz[ids.tolist().index(9)], [0.0, 0.0, 7.5])
    np.testing.assert_array_equal(xyz[ids.tolist().index(12)], [1.0, 2.0, 3.0])


def test_read_points3d_refuses_a_damaged_file(colmap_dir):
    path = colmap_dir / "points3D.bin"
    whole = path.read_bytes()
    path.write_bytes(whole[:-3])
    with pytest.raises(ValueError, match="points3D.bin"):
        build.read_points3d(path)
    path.write_bytes(whole[:-60])                            # cut inside the last point's header
    with pytest.raises(ValueError, match="ends inside point"):
        build.read_points3d(path)
    path.write_bytes(whole + b"\x01")
    with pytest.raises(ValueError, match="not at the end of the file"):
        build.read_points3d(path)


def test_pack_observations_concatenates_in_row_order():
    observed = {"b": np.array([7, 8], dtype=np.int64), "a": np.array([1, 2, 3], dtype=np.int64),
                "c": np.empty(0, dtype=np.int64)}
    starts, point_ids = build.pack_observations(observed, ["a", "c", "missing", "b"])
    assert starts.tolist() == [0, 3, 3, 3, 5] and starts.dtype == np.int64
    assert point_ids.tolist() == [1, 2, 3, 7, 8] and point_ids.dtype == np.int64
    starts, point_ids = build.pack_observations(observed, [])
    assert starts.tolist() == [0] and point_ids.tolist() == []


def test_shared_point_counts_are_set_intersections():
    rng = np.random.default_rng(0)
    tracks = [rng.choice(300, size=rng.integers(0, 120), replace=False) * 7 + 3 for _ in range(12)]
    tracks[4] = np.concatenate([tracks[4], tracks[4][:10]])   # a point seen through two keypoints counts once
    tracks[7] = np.empty(0, dtype=np.int64)                   # a view that sees nothing
    starts, point_ids = build.pack_observations(dict(enumerate(tracks)), range(12))
    counts = build.shared_point_counts(starts, point_ids)
    assert counts.dtype == np.uint16 and counts.shape == (12, 12)
    np.testing.assert_array_equal(counts, shared_counts([set(track.tolist()) for track in tracks]))
    np.testing.assert_array_equal(np.diag(counts), [len(set(track.tolist())) for track in tracks])


def test_shared_point_counts_saturate_instead_of_wrapping():
    many = np.arange(70_000, dtype=np.int64)
    starts, point_ids = build.pack_observations({0: many, 1: many, 2: many[:5]}, range(3))
    assert build.shared_point_counts(starts, point_ids).tolist() == [[65535, 65535, 5], [65535, 65535, 5], [5, 5, 5]]


def test_load_sparse_points_parses_once_then_reads_its_own_file(colmap_dir, tmp_path):
    names = [IMAGES[4][0], IMAGES[0][0], IMAGES[2][0]]
    cache = tmp_path / "sparse_points.npz"
    points = build.load_sparse_points(colmap_dir, cache, names)
    assert points["view_names"].tolist() == names
    assert points["p3d_ids"].tolist() == [1, 2, 3, 5, 9, 12]
    assert points["view_starts"].tolist() == [0, 2, 6, 6] and points["view_pid"].tolist() == [12, 3, 5, 9, 5, 12]
    assert not cache.exists()                                # writing the file is the caller's business
    np.savez(cache, **points)

    # With the file in place the COLMAP model is not needed, for the same views or for some of them in any order.
    nowhere = tmp_path / "no colmap here"
    again = build.load_sparse_points(nowhere, cache, names)
    assert all(np.array_equal(again[key], points[key]) for key in points)
    fewer = build.load_sparse_points(nowhere, cache, [IMAGES[0][0], IMAGES[4][0]])
    assert fewer["view_starts"].tolist() == [0, 4, 6] and fewer["view_pid"].tolist() == [5, 9, 5, 12, 12, 3]
    np.testing.assert_array_equal(fewer["p3d_xyz"], points["p3d_xyz"])

    # A view the file lacks sends it back to the COLMAP model.
    more = names + [IMAGES[1][0]]
    with pytest.raises(FileNotFoundError):
        build.load_sparse_points(nowhere, cache, more)
    assert build.load_sparse_points(colmap_dir, cache, more)["view_pid"].tolist() == [12, 3, 5, 9, 5, 12, 1, 2, 3]


def test_load_sparse_points_refuses_views_and_points_the_model_lacks(colmap_dir, tmp_path):
    with pytest.raises(ValueError, match="1 views are not in"):
        build.load_sparse_points(colmap_dir, tmp_path / "none.npz", [IMAGES[0][0], "not in the model.jpg"])
    write_points3d_bin(colmap_dir / "points3D.bin", [point for point in POINTS if point[0] != 9])
    with pytest.raises(ValueError, match="not in"):
        build.load_sparse_points(colmap_dir, tmp_path / "none.npz", [IMAGES[0][0]])


# --- summary --------------------------------------------------------------------------
def test_period_names():
    assert build.period_names((1859.0, 1970.0, 2019.2877)) == ["before 1859", "1859 to 1970", "1970 to 2019.2877",
                                                              "from 2019.2877"]


def test_covisible_medians_count_usable_views_in_and_outside_the_period():
    period = np.array([0, 0, 0, 1, 1, 2])
    usable = np.array([True, True, False, True, True, False])
    covis = np.zeros((6, 6), dtype=np.uint16)
    for i, j, count in [(0, 1, 50), (0, 2, 100), (0, 3, 40), (0, 4, 10), (1, 3, 30), (1, 4, 35), (3, 4, 5), (0, 5, 99)]:
        covis[i, j] = covis[j, i] = count
    np.fill_diagonal(covis, 500)
    medians = build.covisible_medians(covis, period, usable, 3)
    # View 0 is linked to 1 (its period) and 3; view 1 to 0 and to 3 and 4 (30 points are enough);
    # views 2 and 5 are not usable and count for nobody; views 3 and 4 share only 5 points.
    assert medians["same_period"].tolist()[:2] == [1.0, 0.0]
    assert medians["other_periods"].tolist()[:2] == [1.5, 1.5]
    assert medians.iloc[2].isna().all()                     # no usable view in the last period


def test_summary_table_rows_add_up():
    views, thumbs, shared = synthetic_table()
    flagged = build.flag_views(views, thumbs, shared, FIXES)
    table = build.summary_table(flagged, shared, (1970.0, FIXES.after_fire_from))
    assert table.index.tolist() == ["before 1970", "1970 to 2019.2877", "from 2019.2877", "all"]
    assert table["views"].tolist() == [1, 7, 1, 9]
    assert table["usable"].tolist() == [1, 2, 0, 3]
    by_reason = dict(zip(build.EXCLUSIONS, table.loc["1970 to 2019.2877", list(build.EXCLUSIONS)].tolist()))
    assert by_reason == {"misdated": 0, "montage": 0, "fire_day": 1, "uncertain_date": 0, "before_spire": 0,
                         "rolled": 2, "duplicate": 1, "bad_distortion": 1}
    assert table.loc["from 2019.2877", "misdated"] == 1     # sideways too, but it counts once
    reasons = table[list(build.EXCLUSIONS)].sum(axis=1)
    assert (table["views"] == reasons + table["usable"]).all()
    assert {"same_period", "other_periods"} <= set(table.columns)


# --- building -------------------------------------------------------------------------
def test_write_together_changes_the_files_only_when_every_writer_finishes(tmp_path):
    numbers, table = tmp_path / "numbers.npy", tmp_path / "table.parquet"
    build.write_together({numbers: lambda path: np.save(path, np.arange(3)),
                          table: lambda path: pd.DataFrame({"a": [1]}).to_parquet(path)})
    assert np.load(numbers).tolist() == [0, 1, 2] and pd.read_parquet(table)["a"].tolist() == [1]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["numbers.npy", "table.parquet"]

    def interrupted(path: Path) -> None:
        raise KeyboardInterrupt

    # The first file was written again, the second was not: neither of the old files is replaced.
    with pytest.raises(KeyboardInterrupt):
        build.write_together({numbers: lambda path: np.save(path, np.arange(5)), table: interrupted})
    assert np.load(numbers).tolist() == [0, 1, 2] and pd.read_parquet(table)["a"].tolist() == [1]


def test_renumbered_rows():
    old = ["a", "b", "c", "d"]
    assert build.renumbered_rows(old, old).tolist() == []
    assert build.renumbered_rows(old, old + ["e", "f"]).tolist() == []                  # rows added at the end
    assert build.renumbered_rows(old, ["a", "x", "b", "c", "d"]).tolist() == [1, 2, 3]  # a view in between
    assert build.renumbered_rows(old, ["a", "c", "d"]).tolist() == [1, 2, 3]            # a view gone
    assert build.renumbered_rows(old, ["a", "b", "c"]).tolist() == [3]
    assert build.renumbered_rows([], ["a"]).tolist() == []


@pytest.fixture
def tiny_scene(tmp_path, monkeypatch, colmap_dir) -> dict:
    """A scene of eight views for ``main()``: tables, images and COLMAP files on disk, and a fake download.

    View 0 is a stereo card, 3 is a modern view outside the first table, 5 is dated on the day of the
    fire, 6 is misdated, and the download of the views in the returned ``failing`` set fails.
    """
    dates = [1900.5, 1950.5, 2000.5, 2005.5, 2010.5, 2019.2860, 2020.5, 2021.5]
    sizes = [(200, 100)] + [(96, 64)] * 7
    tracks = [[1, 2, 3], [1, 2, 3, 5], [5, 9, 12], [1], [9, 12, -1, 3], [2], [1, 9], [3, 5]]
    records = []
    for i, (date, (width, height)) in enumerate(zip(dates, sizes)):
        yaw, centre = on_circle(i, n_places=8)
        qvec, tvec = c2w_to_colmap(camera(yaw, centre=centre))
        picture(i, (width, height)).convert("RGB").save(tmp_path / f"view{i}.jpg")
        records.append({"scene_id": 25512, "model": 0, "file_key": f"view{i}.jpg",
                        "path": f"commons/Sub {i}/0/pictures/view{i}.jpg", "title": f"View {i}",
                        "date_lo": date - 0.001, "date_hi": date + 0.001, "date_mid": date, "k1": 0.0, "k2": 0.0,
                        **dict(zip(("qw", "qx", "qy", "qz"), qvec)), **dict(zip(("tx", "ty", "tz"), tvec)),
                        "image_path": str(tmp_path / f"view{i}.jpg"), "image_width": width, "image_height": height})
    full = pd.DataFrame(records)
    # The COLMAP model has one more image than the scene has views, 30 units from the middle of the circle.
    poses = [(row[["qw", "qx", "qy", "qz"]].tolist(), row[["tx", "ty", "tz"]].tolist()) for _, row in full.iterrows()]
    write_images_bin(colmap_dir / "images.bin", list(zip(full["path"], tracks)) + [("not a view.jpg", [1, 2])],
                     poses + [c2w_to_colmap(camera(centre=(30.0, 0.0, 0.0)))])
    full.iloc[[2, 4]][["file_key"]].to_parquet(tmp_path / "base.parquet", index=False)
    (tmp_path / "fixes.json").write_text(json.dumps({
        "scene_id": 25512, "fire_start": 2019.2849, "after_fire_from": 2019.2877, "misdated": {"view6.jpg": "why"}}))
    failing = {"view7.jpg"}
    image_columns = ["image_path", "image_width", "image_height"]

    def fake_download(bucket, paths, views, max_side, workers):
        """Like ``download_view_images``: drops the views that fail and adds the image columns."""
        found = views[~views["file_key"].isin(failing)].merge(full[["file_key"] + image_columns], on="file_key")
        found.attrs["failed"] = [{"file_key": key, "error": "HTTPError()"} for key in views["file_key"]
                                 if key in failing]
        return found

    monkeypatch.setattr(build, "build_view_table", lambda paths, scene, model: full.drop(columns=image_columns))
    monkeypatch.setattr(build, "download_view_images", fake_download)
    monkeypatch.setattr(build, "MegaScenesBucket", lambda: None)
    out = tmp_path / "dataset"
    arguments = ["build_dataset.py", "--base-table", str(tmp_path / "base.parquet"), "--date-fixes",
                 str(tmp_path / "fixes.json"), "--out", str(out), "--colmap-dir"]
    return {"out": out, "arguments": arguments, "colmap_dir": colmap_dir, "failing": failing, "tracks": tracks}


def read_folder(out: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    with np.load(out / "sparse_points.npz") as archive:
        return pd.read_parquet(out / "views.parquet"), np.load(out / "covisibility.npy"), dict(archive)


def test_main_writes_a_consistent_folder_and_a_rerun_needs_no_colmap_files(tiny_scene, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", tiny_scene["arguments"] + [str(tiny_scene["colmap_dir"])])
    build.main()
    views, covis, points = read_folder(tiny_scene["out"])
    # View 0 is a stereo card, 3 is neither old nor after the fire nor in the first table, 7 failed to download.
    assert views["file_key"].tolist() == ["view1.jpg", "view2.jpg", "view4.jpg", "view5.jpg", "view6.jpg"]
    assert views["excluded"].tolist() == ["", "", "", "fire_day", "misdated"]
    assert views["usable"].tolist() == [True, True, True, False, False]
    assert not views["rolled"].any() and (views["duplicate_of"] == -1).all()
    assert points["view_names"].tolist() == views["path"].tolist()
    kept = [set(tiny_scene["tracks"][i]) - {-1} for i in (1, 2, 4, 5, 6)]
    assert points["view_starts"].tolist() == np.cumsum([0] + [len(track) for track in kept]).tolist()
    np.testing.assert_array_equal(covis, shared_counts(kept))
    assert sorted(path.name for path in tiny_scene["out"].iterdir()) == ["covisibility.npy", "scene.json",
                                                                         "sparse_points.npz", "views.parquet"]
    # scene.json holds the box of ALL nine registered cameras: the eight on the circle of radius 10 and the far one.
    scene = json.loads((tiny_scene["out"] / "scene.json").read_text())
    assert set(scene) == {"scene_id", "model", "n_registered_cameras", "camera_min", "camera_max", "camera_extent"}
    assert (scene["scene_id"], scene["model"], scene["n_registered_cameras"]) == (25512, 0, 9)
    np.testing.assert_allclose(scene["camera_min"], [-10.0, -10.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(scene["camera_max"], [30.0, 10.0, 0.0], atol=1e-9)
    assert scene["camera_extent"] == pytest.approx(40.0)
    printed = capsys.readouterr().out
    assert "download failed: view7.jpg" in printed and "1 downloads failed" in printed
    assert "5 views; 3 usable" in printed and "9 registered cameras" in printed
    assert "rows kept" not in printed                        # there was no table to keep the rows of

    # A second run finds what it needs in sparse_points.npz and gives the same folder.
    monkeypatch.setattr(sys, "argv", tiny_scene["arguments"] + [str(tmp_path / "no colmap here")])
    build.main()
    again_views, again_covis, again_points = read_folder(tiny_scene["out"])
    pd.testing.assert_frame_equal(again_views, views)
    np.testing.assert_array_equal(again_covis, covis)
    assert all(np.array_equal(again_points[key], points[key]) for key in points)
    assert json.loads((tiny_scene["out"] / "scene.json").read_text()) == scene
    assert "rows kept: all 5 rows of the existing table keep their view (0 rows added" in capsys.readouterr().out

    # When the failed download succeeds later, the new view is parsed from the COLMAP model and added.
    tiny_scene["failing"].clear()
    with pytest.raises(FileNotFoundError):
        build.main()
    monkeypatch.setattr(sys, "argv", tiny_scene["arguments"] + [str(tiny_scene["colmap_dir"])])
    build.main()
    views, covis, points = read_folder(tiny_scene["out"])
    assert views["file_key"].tolist()[-1] == "view7.jpg" and covis.shape == (6, 6)
    assert points["view_pid"][points["view_starts"][5]:].tolist() == [3, 5]
    assert "keep their view (1 rows added" in capsys.readouterr().out   # a view added at the end moves no row


def test_main_refuses_to_renumber_the_rows_of_an_existing_table(tiny_scene, monkeypatch, tmp_path):
    """Depth maps are stored by row: a rebuild that would give a row another view needs ``--renumber``."""
    monkeypatch.setattr(sys, "argv", tiny_scene["arguments"] + [str(tiny_scene["colmap_dir"])])
    build.main()
    before = read_folder(tiny_scene["out"])[0]
    assert before["file_key"].tolist() == ["view1.jpg", "view2.jpg", "view4.jpg", "view5.jpg", "view6.jpg"]
    # The first table now has view 3 as well: it would become row 2 and move the three views after it.
    pd.DataFrame({"file_key": ["view2.jpg", "view3.jpg", "view4.jpg"]}).to_parquet(tmp_path / "base.parquet")
    with pytest.raises(ValueError, match=r"3 of its 5 rows .* row 2, 'view4.jpg', which would become 'view3.jpg'"):
        build.main()
    pd.testing.assert_frame_equal(read_folder(tiny_scene["out"])[0], before)    # nothing was replaced
    monkeypatch.setattr(sys, "argv", sys.argv + ["--renumber"])
    build.main()
    assert read_folder(tiny_scene["out"])[0]["file_key"].tolist() == [
        "view1.jpg", "view2.jpg", "view3.jpg", "view4.jpg", "view5.jpg", "view6.jpg"]


def test_main_applies_every_rule_of_the_fixes_file(tiny_scene, monkeypatch, tmp_path):
    fixes = {"scene_id": 25512, "fire_start": 2019.2849, "after_fire_from": 2019.2877,
             "misdated": {"view6.jpg": "why"}, "exclude_before": 1960, "duplicates": [["view2.jpg", "view4.jpg"]]}
    (tmp_path / "fixes.json").write_text(json.dumps(fixes))
    monkeypatch.setattr(sys, "argv", tiny_scene["arguments"] + [str(tiny_scene["colmap_dir"])])
    build.main()
    views = read_folder(tiny_scene["out"])[0]
    assert views["file_key"].tolist() == ["view1.jpg", "view2.jpg", "view4.jpg", "view5.jpg", "view6.jpg"]
    assert views["excluded"].tolist() == ["before_spire", "", "duplicate", "fire_day", "misdated"]    # view 1: 1950
    assert views["duplicate_of"].tolist() == [-1, -1, 1, -1, -1]
    # A name that is no view of the scene is a typing mistake, not a photo to skip silently.
    fixes["montages"] = {"view 6.jpg": "a typo"}
    (tmp_path / "fixes.json").write_text(json.dumps(fixes))
    with pytest.raises(ValueError, match=r"not posed and dated views of the scene: \['view 6.jpg'\]"):
        build.main()
