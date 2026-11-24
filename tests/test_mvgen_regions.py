"""CPU tests for the changed-region masks (``timemachine.mvgen.regions``) on a synthetic scene; no real data.

The scene (:func:`write_scene`) is a box-shaped building with a spire, photographed from a ring of
cameras, 72 dated before the edge and 48 after it:

* the *box* (its four walls) is observed by the views of both sides that face a wall;
* the *spire* above it is observed from all around, but only before the edge: it is gone;
* a *sign* on the wall that faces +z is observed only before the edge, and only from that side: gone;
* a *tarp* on the wall that faces +x is observed only after the edge: new;
* one *unlucky* point on the wall that faces -x has three observers before the edge and none after
  it, as a point of the box can have by chance.

Every spire point is drawn as a dark dot into every view's (white) image file, so that the masks
can be compared with the images that ``MVGenScene.make_sample`` returns, pixel for pixel. Three
more views stand apart from the ring: one that looks away from the building, one that is not
usable, and one with a distorted lens whose image file is half the size of its camera.

Run from MVGenMaster's checkout, in its environment (see ``timemachine/mvgen/__init__.py``).
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

my_diffusers = pytest.importorskip("my_diffusers", reason="runs in MVGenMaster's environment, from its checkout")

# Under pytest the repo root is on the import path (its conftest.py), so ``src`` is the repo's own ``src``
# package and hides MVGenMaster's ``src`` folder, which ``timemachine.mvgen.probes`` (used by the overlay
# script) imports from. Let the package search that folder too.
import src  # noqa: E402

MVGEN_SRC = str(Path(my_diffusers.__file__).resolve().parents[1] / "src")
if MVGEN_SRC not in list(src.__path__):
    src.__path__.append(MVGEN_SRC)

from PIL import Image  # noqa: E402

from timemachine.data.cameras import c2w_to_colmap  # noqa: E402
from timemachine.data.dataset import crop_intrinsics, load_view  # noqa: E402
from timemachine.mvgen import regions as regions_module  # noqa: E402
from timemachine.mvgen.data import MVGenScene, depth_path  # noqa: E402
from timemachine.mvgen.regions import (  # noqa: E402
    FIRE, RULES, ChangedRegions, RegionConfig, changed_points, hidden_behind, in_full_image, latent_weights,
    load_sparse, main_groups, observation_pairs, seen_from_behind, spacing, view_sides, widest_cosine,
)

REPO = Path(__file__).resolve().parents[1]
RES = 256                                   # side of the crops (the pixel sizes of RegionConfig are halved)
N_BEFORE, N_AFTER = 72, 48                  # ring views dated before and after the edge
AWAY, UNUSABLE, DISTORTED = 120, 121, 122   # rows of the three views that stand apart
N = 123
FRONT, SIDE, BACK = 0, 18, 36               # before-views that face the +z wall, the +x wall and the -z wall
EYE_HEIGHT, RING_RADIUS, LOOK_AT = -0.8, 6.0, np.array([0.0, -1.1, 0.0])     # y points down; the ground is y = 0
OBSERVED = 0.6                              # the chance that a view which faces a point observes it


# ----------------------------------------------------------------------------- the synthetic scene
def look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world of a level camera at ``centre`` that looks at ``target`` (x right, y down, z forward)."""
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, np.cross(forward, right), forward, centre
    return c2w


def ring_position(angle: float) -> np.ndarray:
    return np.array([RING_RADIUS * np.sin(angle), EYE_HEIGHT, RING_RADIUS * np.cos(angle)])


def scene_points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The points of the scene: ``[P, 3]`` positions, ``[P]`` labels, and ``[P, 3]`` the outward normal of the
    wall a point is on (zero for the spire, which shows from every side)."""
    xyz, label, normal = [], [], []
    for n in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0]):                # the four walls, 2 wide and 1 high
        n = np.array(n, dtype=float)
        along = np.array([n[2], 0.0, -n[0]])
        a, height = np.meshgrid(np.linspace(-0.95, 0.95, 16), np.linspace(-0.95, -0.05, 8))
        wall = n + a.reshape(-1, 1) * along + height.reshape(-1, 1) * np.array([0.0, 1.0, 0.0])
        xyz.append(wall + rng.normal(0, 0.005, wall.shape))
        label += ["box"] * len(wall)
        normal.append(np.tile(n, (len(wall), 1)))
    spire = np.column_stack([rng.uniform(-0.03, 0.03, 60), np.linspace(-2.2, -1.2, 60), rng.uniform(-0.03, 0.03, 60)])
    sign = np.column_stack([rng.uniform(-0.2, 0.2, 60), rng.uniform(-0.7, -0.5, 60), np.full(60, 1.02)])
    tarp = np.column_stack([np.full(60, 1.02), rng.uniform(-0.7, -0.5, 60), rng.uniform(-0.2, 0.2, 60)])
    unlucky = np.array([[-1.02, -0.5, 0.03]])
    xyz += [spire, sign, tarp, unlucky]
    label += ["spire"] * 60 + ["sign"] * 60 + ["tarp"] * 60 + ["unlucky"]
    normal += [np.zeros((60, 3)), np.tile([0.0, 0.0, 1.0], (60, 1)), np.tile([1.0, 0.0, 0.0], (60, 1)),
               np.array([[-1.0, 0.0, 0.0]])]
    return np.concatenate(xyz), np.array(label), np.concatenate(normal)


def file_pixels(xyz: np.ndarray, c2w: np.ndarray, camera: dict) -> np.ndarray:
    """``[P, 2]`` where a view's image *file* shows the points (pixel centres at 0.5), by COLMAP's lens model;
    NaN for points behind the camera."""
    cam = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    stretch = 1 + camera["k1"] * (x * x + y * y)
    u = (camera["fx"] * x * stretch + camera["cx"]) * camera["image_width"] / camera["camera_width"]
    v = (camera["fy"] * y * stretch + camera["cy"]) * camera["image_height"] / camera["camera_height"]
    return np.where((cam[:, 2] > 0)[:, None], np.column_stack([u, v]), np.nan)


def write_scene(root: Path, seed: int = 0) -> SimpleNamespace:
    """Write the scene of the module docstring as a dataset folder and return what the tests need to know."""
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    (root / "depth").mkdir()
    xyz, label, normal = scene_points(rng)

    angles = np.concatenate([2 * np.pi * np.arange(N_BEFORE) / N_BEFORE,
                             2 * np.pi * (np.arange(N_AFTER) + 0.5) / N_AFTER, [0.0, 0.3, 0.2]])
    c2w = np.stack([look_at(ring_position(angle), LOOK_AT) for angle in angles])
    c2w[AWAY] = look_at(ring_position(0.0), 2 * ring_position(0.0) - LOOK_AT)
    date_mid = np.concatenate([2010 + 0.1 * np.arange(N_BEFORE), 2020 + 0.05 * np.arange(N_AFTER),
                               [2015.03, 2016.03, 2018.53]])
    usable = np.arange(N) != UNUSABLE
    plain = dict(camera_width=384, camera_height=256, fx=300.0, fy=300.0, cx=192.0, cy=128.0, k1=0.0, k2=0.0,
                 image_width=384, image_height=256)
    cameras = [dict(plain) for _ in range(N)]
    cameras[DISTORTED].update(camera_width=768, camera_height=512, fx=600.0, fy=600.0, cx=384.0, cy=256.0, k1=-0.12)

    # Who observes what: a view that faces a point observes it with probability OBSERVED, if the point
    # exists at the view's date.
    before = (date_mid < FIRE) & usable
    after = (date_mid >= FIRE) & usable
    to_camera = c2w[:, None, :3, 3] - xyz[None]
    facing = (to_camera * normal[None]).sum(axis=2) > 0.25 * np.linalg.norm(to_camera, axis=2)
    facing |= (label == "spire")[None]
    facing[AWAY] = False
    exists = np.where(np.isin(label, ["spire", "sign"])[None], before[:, None],
                      np.where((label == "tarp")[None], after[:, None], True))
    observed = facing & exists & (rng.random(facing.shape) < OBSERVED)
    observed[:, label == "unlucky"] = False
    observed[[53, 54, 55], np.flatnonzero(label == "unlucky")[0]] = True     # three before-views in front of it

    rows = []
    for i in range(N):
        pixels = np.full((256, 384, 3), 255, dtype=np.uint8)
        dots = file_pixels(xyz[label == "spire"], c2w[i], cameras[i])
        for u, v in dots[np.isfinite(dots).all(axis=1)]:
            if 1 <= u < 383 and 1 <= v < 255:
                pixels[int(v) - 1:int(v) + 2, int(u) - 1:int(u) + 2] = 0
        path = root / "images" / f"view_{i:03d}.png"
        Image.fromarray(pixels).save(path)
        q, t = c2w_to_colmap(c2w[i])
        rows.append(dict(file_key=path.name, image_path=str(path), **cameras[i], qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                         tx=t[0], ty=t[1], tz=t[2], date_lo=date_mid[i] - 0.001, date_hi=date_mid[i] + 0.001,
                         date_mid=date_mid[i], usable=bool(usable[i])))
    pd.DataFrame(rows).to_parquet(root / "views.parquet", index=False)

    covis = np.full((N, N), 100, dtype=np.uint16)
    np.fill_diagonal(covis, 400)
    np.save(root / "covisibility.npy", covis)
    ids = 100 + 7 * np.arange(len(xyz))                                     # sorted, with gaps, as COLMAP's ids
    seen = [np.flatnonzero(observed[i]) for i in range(N)]
    np.savez(root / "sparse_points.npz", p3d_ids=ids, p3d_xyz=xyz, view_pid=np.concatenate([ids[s] for s in seen]),
             view_starts=np.concatenate([[0], np.cumsum([len(s) for s in seen])]))
    for row in (FRONT, DISTORTED):                                          # make_sample needs a reference's depth
        np.save(depth_path(root, row), np.full((RES, RES), 5.0, dtype=np.float16))
    (root / "scene.json").write_text(json.dumps({"camera_extent": 2 * RING_RADIUS}))
    return SimpleNamespace(dir=root, xyz=xyz, label=label, c2w=c2w, date_mid=date_mid, usable=usable,
                           observed=observed)


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> SimpleNamespace:
    return write_scene(tmp_path_factory.mktemp("regions") / "scene")


@pytest.fixture(scope="module")
def scene(folder) -> MVGenScene:
    return MVGenScene(folder.dir, resolution=RES, require_depth=False)


@pytest.fixture(scope="module")
def regions(scene) -> ChangedRegions:
    return ChangedRegions(scene)


def labels_of(folder: SimpleNamespace, xyz: np.ndarray) -> np.ndarray:
    """The labels of scene points given by their positions."""
    by_position = {tuple(point): name for point, name in zip(folder.xyz, folder.label)}
    return np.array([by_position[tuple(point)] for point in xyz], dtype=object)


def points_of(folder: SimpleNamespace, cfg: RegionConfig = RegionConfig(), dates: np.ndarray | None = None):
    """``changed_points`` on the scene's arrays (the scene object is not needed for it)."""
    views = pd.read_parquet(folder.dir / "views.parquet")
    side = view_sides(folder.date_mid, folder.usable, FIRE)
    return changed_points(load_sparse(folder.dir), views, folder.c2w, folder.date_mid if dates is None else dates,
                          side, cfg)


def sample_pixels(scene: MVGenScene, row: int, xyz: np.ndarray) -> np.ndarray:
    """``[P, 2]`` integer (column, line) of the pixels where ``make_sample``'s cameras put the points."""
    sample = scene.make_sample(np.array([row]), 1)
    K, w2c = sample["K"][0].numpy().astype(np.float64), sample["w2c"][0].numpy().astype(np.float64)
    cam = (xyz * sample["scale"]) @ w2c[:3, :3].T + w2c[:3, 3]             # the sample's cameras are rescaled
    return np.floor((cam / cam[:, 2:3]) @ K.T)[:, :2].astype(int)


def sample_pixels_without_depth(scene: MVGenScene, row: int, xyz: np.ndarray) -> np.ndarray:
    """As :func:`sample_pixels`, for views that have no depth file (``make_sample`` needs one): the same
    camera, taken from ``load_view`` and the scene's poses as ``make_sample`` takes it."""
    K = load_view(next(scene.views.iloc[[row]].itertuples()), RES)[1]
    w2c = np.linalg.inv(scene.c2ws[row])
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    return np.floor((cam / cam[:, 2:3]) @ K.T)[:, :2].astype(int)


def load_script(name: str):
    """A script of ``scripts/mvgen`` as a module."""
    spec = importlib.util.spec_from_file_location(f"mvgen_{name}_script", REPO / "scripts" / "mvgen" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- small parts
def test_view_sides_are_the_usable_views_of_the_two_periods_next_to_the_edge():
    dates = np.array([1900.0, 1969.9, 1970.0, 2019.28, 2019.2877, 2021.0, 2030.0, 2015.0])
    usable = np.array([True, True, True, True, True, True, True, False])
    assert view_sides(dates, usable, 2019.2877, (1970.0, 2019.2877)).tolist() == [-1, -1, 0, 0, 1, 1, 1, -1]
    assert view_sides(dates, usable, 2019.2877, (1970.0, 2019.2877, 2025.0)).tolist() == [-1, -1, 0, 0, 1, 1, -1, -1]
    assert view_sides(dates, usable, 1970.0, (1970.0, 2019.2877)).tolist() == [0, 0, 1, 1, -1, -1, -1, -1]
    # an edge that is not one of the scene's still has a period on either side; without edges both are open
    assert view_sides(dates, usable, 2000.0, (1970.0, 2019.2877)).tolist() == [-1, -1, 0, 1, -1, -1, -1, -1]
    assert view_sides(dates, usable, 2000.0).tolist() == [0, 0, 0, 1, 1, 1, 1, -1]


def test_observation_pairs_count_a_point_once_per_view():
    sparse = {"p3d_ids": np.array([10, 13, 16]), "view_starts": np.array([0, 3, 3, 5]),
              "view_pid": np.array([16, 10, 16, 13, 10])}                   # view 0 sees point 16 twice; view 1 nothing
    view, point = observation_pairs(sparse)
    assert list(zip(view.tolist(), point.tolist())) == [(0, 0), (0, 2), (2, 0), (2, 1)]
    with pytest.raises(ValueError, match="not in p3d_ids"):
        observation_pairs({**sparse, "view_pid": np.array([16, 10, 16, 13, 11])})


def test_in_full_image_follows_the_lens_model_only_where_it_is_valid():
    camera = dict(fx=100.0, fy=100.0, cx=50.0, cy=40.0, k1=0.0, k2=0.0, camera_width=100, camera_height=80)
    cam = np.array([[0.0, 0.0, 2.0], [0.98, 0.0, 2.0], [1.02, 0.0, 2.0], [0.0, -0.82, 2.0], [0.0, 0.0, -2.0]])
    assert in_full_image(cam, camera).tolist() == [True, True, False, False, False]
    # A barrel lens shows more than the pinhole camera with its focal length ...
    barrel = {**camera, "k1": -0.2}
    assert in_full_image(cam, barrel).tolist() == [True, True, True, True, False]
    # ... but its polynomial turns back at r^2 = 1 / (3 * 0.2): a ray far outside the field of view, which the
    # formula alone would put at pixel 50 + 100 * 2.2 * (1 - 0.2 * 4.84) = 57, is not in the image.
    assert not in_full_image(np.array([[2.2, 0.0, 1.0]]), barrel)[0]


def test_crop_intrinsics_are_those_of_load_view(scene):
    for row in (FRONT, AWAY, DISTORTED):
        view = next(scene.views.iloc[[row]].itertuples())
        for size in (RES, 128):
            np.testing.assert_array_equal(crop_intrinsics(view, size), load_view(view, size)[1])
    # the distorted view's file is half the size of its camera: the crop has half the camera's focal length
    assert crop_intrinsics(next(scene.views.iloc[[DISTORTED]].itertuples()), RES)[0].tolist() == [300.0, 0.0, 128.0]


def test_spacing_and_main_groups():
    line = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
    assert spacing(line, 1).tolist() == [1.0] * 6
    assert spacing(line, 2).tolist() == [2.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    assert spacing(line[:1], 3).tolist() == [0.0] and len(spacing(line[:0], 3)) == 0
    assert spacing(line[:3], 5).tolist() == [2.0, 1.0, 2.0]                 # fewer points than neighbours: the farthest

    # a chain is one group when its steps are within the link; a far, smaller cluster is another
    cluster = np.array([[50.0, 0.0, 0.0], [50.5, 0.0, 0.0]])
    points = np.concatenate([line, cluster])
    assert main_groups(points, 1.1, 0.5).tolist() == [True] * 6 + [False] * 2
    assert main_groups(points, 1.1, 0.25).all()                             # both hold a quarter of the points
    # with a shorter link the chain falls apart into single points, and only the pair is a group of a fifth
    assert main_groups(points, 0.9, 0.2).tolist() == [False] * 6 + [True] * 2
    assert main_groups(points[:0], 1.0, 0.2).shape == (0,)


def test_widest_cosine_and_seen_from_behind():
    cos30 = np.cos(np.pi / 6)
    np.testing.assert_allclose(widest_cosine(np.array([1.0, 0.5, 0.0]), 30.0), [cos30, 0.5 * cos30 - 0.5, -1.0])

    def behind(seen_from: list[float], degrees: float) -> bool:
        """Whether a camera ``degrees`` away from the direction +z is behind a point with that ``seen_from``."""
        camera = 7.0 * np.array([[np.sin(np.radians(degrees)), 0.0, np.cos(np.radians(degrees))]])
        return bool(seen_from_behind(camera, np.array([seen_from]))[0])

    one_side = [0.0, 0.0, 1.0]                                              # every observer stood in direction +z
    assert [behind(one_side, angle) for angle in (0, 80, 100, 180)] == [False, False, True, True]
    spread = [0.0, 0.0, 0.5]                                                # observers spread around +z: 120 degrees
    assert [behind(spread, angle) for angle in (100, 115, 125, 180)] == [False, False, True, True]
    around = [0.0, 0.01, 0.0]                                               # observers all around: no back
    assert not any(behind(around, angle) for angle in (0, 90, 180))


def test_hidden_behind_needs_nearer_occluders_all_around_the_point():
    point, depth = np.array([[10.0, 10.0]]), np.array([5.0])
    around = point + 2.0 * np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])   # one occluder in each quadrant

    def hidden(occluder_depths, reach=3.0, margin=1.0, occluders=around, count=8) -> bool:
        return bool(hidden_behind(point, depth, occluders, np.asarray(occluder_depths, dtype=float), count, reach,
                                  margin)[0])

    assert hidden([2, 2, 2, 2])
    assert not hidden([2, 2, 2, 9])                                         # nothing in front on one side
    assert not hidden([4.5, 4.5, 4.5, 4.5]) and hidden([4.5, 4.5, 4.5, 4.5], margin=0.25)     # nearer than the margin
    assert not hidden([2, 2, 2, 2], reach=2.5)                              # none within reach
    assert not hidden([2, 2, 2, 2], occluders=point + 2.0 * np.array([[1, 1], [-1, 1], [1, 0.5], [-1, 0.5]]))
    assert not hidden([], occluders=np.zeros((0, 2)))
    assert hidden_behind(np.zeros((0, 2)), np.zeros(0), around, np.full(4, 2.0), 8, 3.0, 1.0).shape == (0,)
    # Only the nearest ``count`` occluders are looked at: four far ones behind the point, nearer in the image,
    # use them up.
    crowd = np.concatenate([point + 0.5 * np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]]), around])
    assert hidden([9, 9, 9, 9, 2, 2, 2, 2], occluders=crowd)
    assert not hidden([9, 9, 9, 9, 2, 2, 2, 2], occluders=crowd, count=4)
    # each point is judged by its own neighbours
    points = np.array([[10.0, 10.0], [40.0, 40.0]])
    assert hidden_behind(points, np.array([5.0, 5.0]), around, np.full(4, 2.0), 8, 3.0, 1.0).tolist() == [True, False]


def test_latent_weights_average_the_mask_over_each_latent_cell():
    mask = np.random.default_rng(0).random((64, 48)) < 0.3
    weights = latent_weights(mask)
    assert weights.shape == (8, 6) and weights.dtype == np.float32
    assert weights.min() >= 0 and weights.max() <= 1
    assert weights.sum() * 64 == pytest.approx(mask.sum())
    assert weights[2, 3] == pytest.approx(mask[16:24, 24:32].mean())
    np.testing.assert_array_equal(latent_weights(mask, 1), mask.astype(np.float32))
    assert latent_weights(np.ones((16, 16), dtype=bool), 4).tolist() == [[1.0] * 4] * 4
    with pytest.raises(ValueError, match="cannot be averaged"):
        latent_weights(np.zeros((60, 64), dtype=bool))


@pytest.mark.parametrize("changes", [
    dict(min_days=0), dict(neighbours=0), dict(spacing_neighbour=0), dict(splat_neighbour=0), dict(max_angle=0.0),
    dict(max_angle=181.0), dict(scale_margin=0.9), dict(min_expected=0.0), dict(min_flagged_share=0.0),
    dict(min_flagged_share=1.1), dict(min_group_share=-0.1), dict(link_factor=0.0), dict(min_pixels=0.0),
    dict(min_pixels=5.0, max_pixels=4.0), dict(occluders=0), dict(occluder_pixels=0.0), dict(occluder_margin=-1.0),
    dict(closing_pixels=-1), dict(min_area=-1),
])
def test_config_is_validated(changes):
    with pytest.raises(ValueError):
        RegionConfig(**changes)


# ----------------------------------------------------------------------------- changed points
def test_spire_and_sign_are_gone_the_tarp_is_new_and_the_box_is_neither(folder):
    points = points_of(folder)
    gone, new = labels_of(folder, points.gone), labels_of(folder, points.new)
    assert set(gone) == {"spire", "sign"} and set(new) == {"tarp"}
    for found, name in ((gone, "spire"), (gone, "sign"), (new, "tarp")):
        assert (found == name).sum() >= 0.8 * (folder.label == name).sum()
    # the walls were there throughout: both sides observe (nearly) every point of them, and nothing else
    stable = labels_of(folder, points.stable)
    assert set(stable) == {"box"} and len(stable) >= 0.99 * (folder.label == "box").sum()

    assert set(points.counts) == {"gone", "new"} and all(tuple(count) == RULES for count in points.counts.values())
    for kind, kept in (("gone", points.gone), ("new", points.new)):
        counts = list(points.counts[kind].values())
        assert counts == sorted(counts, reverse=True) and counts[-1] == len(kept)        # each rule only removes
    assert points.counts["gone"]["candidates"] > 500                        # the walls are candidates too
    assert points.xyz.shape == (len(gone) + len(new), 3) and points.seen_from.shape == points.xyz.shape
    np.testing.assert_array_equal(points.xyz[:len(gone)], points.gone)
    assert points.link > 0

    # The sign was observed from the half circle in front of its wall, the spire from all around: the mean
    # of the sign's directions is long and points away from the wall (+z), the spire's is short.
    resultant = np.linalg.norm(points.gone_seen_from, axis=1)
    assert resultant[gone == "sign"].min() > 0.6 and resultant[gone == "spire"].max() < 0.3
    assert (points.gone_seen_from[gone == "sign"][:, 2] > 0.9 * resultant[gone == "sign"]).all()


def test_a_point_without_observers_by_chance_is_not_flagged(folder):
    alone = RegionConfig(neighbours=1, min_group_share=0.0)                 # rules 4 and 5 keep every flagged point
    unlucky = folder.xyz[folder.label == "unlucky"][0]
    assert folder.observed[:N_BEFORE, folder.label == "unlucky"].sum() == 3
    assert not folder.observed[N_BEFORE:, folder.label == "unlucky"].any()

    def is_gone(points) -> bool:
        return bool((points.gone == unlucky).all(axis=1).any())

    # About 13 views before the edge look at it as its three observers do, and 8 views after it: 3 / 13 * 8 is
    # under the 3 observers that must be expected before none means anything.
    assert not is_gone(points_of(folder, alone))
    assert is_gone(points_of(folder, RegionConfig(neighbours=1, min_group_share=0.0, min_expected=1.0)))
    # With a lower bar it is flagged, and then its neighbours hold it back: the wall's points around it have
    # observers after the edge.
    assert not is_gone(points_of(folder, RegionConfig(min_group_share=0.0, min_expected=1.0)))


def test_observers_of_a_single_day_are_not_enough(folder):
    one_day = np.where(folder.date_mid < FIRE, 2012.0, folder.date_mid)     # every view before the edge on one day
    points = points_of(folder, dates=one_day)
    assert points.counts["gone"]["candidates"] == 0 and len(points.gone) == 0
    assert set(labels_of(folder, points.new)) == {"tarp"}                   # the other side is as it was
    assert len(points_of(folder, RegionConfig(min_days=1), dates=one_day).gone) > 80


def test_small_groups_away_from_the_main_change_are_dropped(folder):
    # The spire and the sign hold about 60 gone points each, the tarp 60 new ones. A group needs its share of
    # the points of both kinds, so each holds a third, and none of them holds 45%.
    assert len(points_of(folder, RegionConfig(min_group_share=0.45)).xyz) == 0
    kept = points_of(folder, RegionConfig(min_group_share=0.25))
    assert set(labels_of(folder, kept.gone)) == {"spire", "sign"} and len(kept.new) > 40
    assert kept.counts["gone"]["kept"] == kept.counts["gone"]["clustered"]


# ----------------------------------------------------------------------------- masks
def test_masks_cover_the_spire_in_the_images_of_make_sample(scene, regions, folder):
    kept_spire = regions.points.gone[labels_of(folder, regions.points.gone) == "spire"]
    for row in (FRONT, DISTORTED):
        sample = scene.make_sample(np.array([row]), 1)
        dark = sample["images"][0].mean(dim=0).numpy() < 0                  # the spire's dots in the white image
        mask = regions.mask(row, RES)
        assert mask.shape == (RES, RES) and mask.dtype == bool
        assert dark.sum() > 100 and mask[dark].mean() > 0.9
        assert regions.share(row, RES) == mask.mean() < 0.03                # the spire and the sign, not the frame
        # The mask sits on the dots: its top is the spire's top, and over the spire's height it is centred on them.
        lines = np.flatnonzero(dark.any(axis=1))
        assert abs(int(np.flatnonzero(mask.any(axis=1))[0]) - int(lines[0])) <= 3
        columns = np.arange(RES)[None]
        centre = lambda picture: (picture[lines] * columns).sum() / picture[lines].sum()        # noqa: E731
        assert abs(centre(mask) - centre(dark)) < 1.0
        # ... and it holds every pixel at which the sample's own cameras put a changed point of the spire
        column, line = sample_pixels(scene, row, kept_spire).T
        assert len(kept_spire) > 48 and mask[line, column].all()
        # the walls are not masked: not at their corners, and not under the sign
        column, line = sample_pixels(scene, row, np.array([[-0.9, -0.1, 1.0], [0.9, -0.9, 1.0], [0.0, -0.2, 1.0]])).T
        assert not mask[line, column].any()


def test_a_point_is_not_drawn_in_views_from_behind_it(scene, regions, folder):
    sign = folder.xyz[folder.label == "sign"]
    spire = regions.points.gone[labels_of(folder, regions.points.gone) == "spire"]
    column, line = sample_pixels_without_depth(scene, FRONT, sign).T
    assert regions.mask(FRONT, RES)[line, column].mean() > 0.8              # from the front the sign is masked
    column, line = sample_pixels_without_depth(scene, BACK, sign).T
    assert not regions.mask(BACK, RES)[line, column].any()                  # from behind the building it is not
    for row in (FRONT, SIDE, BACK):                                         # the spire is masked from every side
        column, line = sample_pixels_without_depth(scene, row, spire).T
        assert regions.mask(row, RES)[line, column].all()


def test_the_two_ways_of_projecting_agree(scene, folder):
    points = folder.xyz[folder.label == "spire"]
    np.testing.assert_array_equal(sample_pixels(scene, FRONT, points),
                                  sample_pixels_without_depth(scene, FRONT, points))


def test_mask_is_empty_for_a_view_that_looks_away(regions):
    assert not regions.mask(AWAY, RES).any() and regions.share(AWAY, RES) == 0.0


def test_views_that_are_not_usable_get_masks_too(regions, scene):
    assert regions.side[UNUSABLE] == -1 and not scene.usable[UNUSABLE]
    assert regions.side[:N_BEFORE].tolist() == [0] * N_BEFORE and regions.side[N_BEFORE:120].tolist() == [1] * N_AFTER
    dark = load_view(next(scene.views.iloc[[UNUSABLE]].itertuples()), RES)[0].mean(dim=0).numpy() < 0
    assert dark.sum() > 100 and regions.mask(UNUSABLE, RES)[dark].mean() > 0.9


def test_masks_are_made_once_and_handed_out_as_copies(regions, monkeypatch):
    drawn = []
    draw = regions._draw

    def counted(row: int, resolution: int) -> np.ndarray:
        drawn.append((row, resolution))
        return draw(row, resolution)

    monkeypatch.setattr(regions, "_draw", counted)
    row = 7                                                                 # a view no other test asks for
    first = regions.mask(row, RES)
    first[:] = True                                                         # the caller may write into its copy
    second = regions.mask(row, RES)
    assert drawn == [(row, RES)] and 0 < second.mean() < 0.03
    assert regions.share(row, RES) == second.mean() and drawn == [(row, RES)]
    regions.mask(row, 128)
    assert drawn == [(row, RES), (row, 128)]                                # another resolution is another mask


def test_masks_follow_the_resolution(regions):
    full, half = regions.mask(FRONT, RES), regions.mask(FRONT, 128)
    assert half.shape == (128, 128)
    grown = np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)
    assert (grown & full).sum() > 0.5 * (grown | full).sum()                # the same place, at half the size
    assert np.abs(np.argwhere(grown).mean(axis=0) - np.argwhere(full).mean(axis=0)).max() < 2


def test_no_changed_points_give_empty_masks(scene):
    nothing = ChangedRegions(scene, edge=2030.0)                            # no view is dated after this edge
    assert len(nothing.xyz) == 0 and nothing.points.link == 0.0
    assert nothing.points.counts["new"]["candidates"] == 0
    assert not nothing.mask(FRONT, RES).any()


def test_occluders_hide_changed_points(scene, regions, monkeypatch):
    # The same view with stable points close in front of everything: nothing of the spire is left.
    assert regions.mask(BACK, RES).any()
    hidden = ChangedRegions.__new__(ChangedRegions)
    hidden.__dict__.update({**regions.__dict__, "_masks": {}})
    seen = {}

    def everything_hidden(uv, z, occluder_uv, occluder_z, count, reach, margin):
        seen.update(count=count, reach=reach, margin=margin, occluders=len(occluder_uv))
        return np.ones(len(uv), dtype=bool)

    monkeypatch.setattr(regions_module, "hidden_behind", everything_hidden)
    assert not hidden.mask(BACK, RES).any()
    cfg = regions.config                                                    # pixel sizes are halved at 256
    assert seen == {"count": cfg.occluders, "reach": cfg.occluder_pixels / 2,
                    "margin": cfg.occluder_margin * regions.points.link, "occluders": len(regions.points.stable)}


# ----------------------------------------------------------------------------- the overlay script
def test_show_regions_writes_sheets_and_shares(folder, tmp_path, capsys):
    show = load_script("show_regions")
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([{"refs": ["view_000.png"], "targets": ["view_018.png", "view_080.png"]},
                                {"refs": ["view_018.png"], "targets": ["view_036.png"]}]))
    show.main(["--dataset", str(folder.dir), "--out", str(tmp_path / "sheets"), "--name", "test", "--rows", str(AWAY),
               "--keys-file", str(keys), "--n-ref", "1", "--resolution", str(RES), "--per-sheet", "3", "--thumb", "64"])
    record = json.loads((tmp_path / "sheets" / "test.json").read_text())
    assert [view["row"] for view in record["views"]] == [AWAY, 0, 18, 80, 36]      # each view once, in the order given
    assert [view["side"] for view in record["views"]] == ["before", "before", "before", "after", "before"]
    assert record["views"][3]["file_key"] == "view_080.png" and record["views"][3]["year"] == pytest.approx(2020.4)
    assert record["points"]["gone"]["kept"] > 80 and record["edge"] == FIRE and record["resolution"] == RES
    assert record["views"][0]["share"] == 0.0
    assert all(0 < view["share"] < 0.03 for view in record["views"][1:])
    sheets = sorted((tmp_path / "sheets").glob("test_*.jpg"))
    assert [sheet.name for sheet in sheets] == ["test_0.jpg", "test_1.jpg"]
    assert Image.open(sheets[0]).size == (2 * 64, 3 * (64 + show._LABEL))
    assert Image.open(sheets[1]).size == (2 * 64, 2 * (64 + show._LABEL))
    printed = capsys.readouterr().out
    assert "gone:" in printed and f"row {AWAY} " in printed and "wrote 2 sheets" in printed

    with pytest.raises(SystemExit, match="no views"):
        show.main(["--dataset", str(folder.dir), "--out", str(tmp_path / "none"), "--resolution", str(RES)])


def test_show_regions_tints_the_mask_and_spreads_views_over_directions(regions):
    show = load_script("show_regions")
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    marked = show.overlay(image, mask)
    assert (marked[~mask] == 200).all() and (marked[2, 2] == show.TINT).all()       # the outline in the tint itself
    assert marked[3, 3, 0] > 200 > marked[3, 3, 1]                                  # inside: redder, still the image

    rows = show.spread_rows(regions, 8)
    assert len(rows) == len(set(rows)) == 8
    assert regions.side[rows].tolist() == [0] * 4 + [1] * 4
    centres = regions.scene.c2ws[:, :3, 3]
    for side_rows in (rows[:4], rows[4:]):                                  # a quarter turn apart, all round the ring
        angles = np.sort(np.degrees(np.arctan2(centres[side_rows, 0], centres[side_rows, 2])))
        assert np.diff(angles).min() > 60 and angles[-1] - angles[0] < 300
    assert len(show.spread_rows(regions, 7)) == 7 and show.spread_rows(regions, 0) == []
    everything = show.spread_rows(regions, 1000)                            # no more than there are
    assert AWAY not in everything and UNUSABLE not in everything and len(everything) == N_BEFORE + 1 + N_AFTER
