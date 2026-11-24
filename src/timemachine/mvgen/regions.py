"""Where a scene changed at a period edge: the changed COLMAP points, and a mask of them for any view.

Whole-image scores barely register Notre-Dame's fire, because the spire and the roof are a few
percent of a frame. This module says *where* the scene differs across a period edge (by default the
fire), so that a score can be restricted to that region. Nothing is drawn by hand: the region
follows from the COLMAP points and from which dated views observe them.

**Changed points** (:func:`changed_points`). A point is *gone* if views dated before the edge
observe it and views dated after it do not, although they look at that place; *new* is the reverse.
Only usable views of the two periods next to the edge count (:func:`view_sides`). Counting alone
does not work: half of the points that five views before the fire observe have no observer after
it, most of them on the unchanged facade, because a point is matched by few of the views that show
it. So a point has to pass five rules (written for a gone point: its "own side" is before the edge,
the "other side" after it; for a new point the sides are swapped):

1. *Observed on several days*: views of at least ``min_days`` different days of its own side observe
   it. The pictures one photographer takes on one day match each other easily and are one piece of
   evidence; this also drops what is only there for a day.
2. *Views that look at it as its observers do*: the views in whose full image it projects, that
   resolve it as finely as its observers (focal length over distance within the observers' range,
   widened by ``scale_margin``) and see it from a direction within ``max_angle`` of the observers'
   mean direction (:func:`widest_cosine`: from farther round, if the observers themselves stood
   in many directions). Views from the far side of the building, or that show the place much
   smaller or larger than any observer, say nothing about the point.
3. *Expected and missing*: its observation rate is the share of such views of its own side that
   observe it; the expected number of observers on the other side is that rate times the number of
   such views there. A point is *testable* if that number is at least ``min_expected``, and
   *flagged* if it is testable and no view of the other side observes it.
4. *Flagged neighbourhood*: rule 3 assumes that views observe a point independently of each other,
   and they do not, so on unchanged surfaces about a fifth of the testable points are flagged by
   chance. What tells a changed surface is that *nearly all* testable points on it are flagged. A
   flagged point is kept if at least ``min_flagged_share`` of its ``neighbours`` nearest testable
   points of the same kind (itself included) are flagged too.
5. *Part of the main change*: what is left is the change itself and smaller places that differ
   for other reasons: foliage (never matched across years), things at ground level that hoardings
   and site cabins hide after the fire. The points of both kinds are grouped, two points belonging
   together if they are closer than ``link_factor`` times the typical spacing of the points, and
   only groups that hold at least ``min_group_share`` of all the points are kept. At Notre-Dame
   that is one group: the spire, the roofs and the scaffolding at the crossing.

The thresholds, and what was measured on Notre-Dame to choose them, are with :class:`RegionConfig`.

**Masks** (:class:`ChangedRegions`). The changed points of both kinds are projected into a view with
the camera of the crop that ``MVGenScene.make_sample`` returns for it, so mask and image align pixel
for pixel. Each point is drawn as a disc whose radius follows the local point density: in 3D it is
the distance to the point's ``splat_neighbour``-th nearest changed point (the dense spire gets small
discs, the sparse roof larger ones that meet), projected by focal length over depth. The discs are
then closed over small gaps and specks are removed. Two kinds of points are left out of a view:

* points it looks at *from behind*: from the half of space their observers were not in
  (:func:`seen_from_behind`). Points on the back of the west gable, which showed when the roof was
  gone, are not drawn on the facade in views from the front;
* points it has something *in front of* (:func:`hidden_behind`): where *stable* points (those that
  both sides observe, so they were there throughout) that are clearly nearer surround a point in
  the image, as the west facade does with the roof in frontal views. The stored depth maps are
  not used for this: they are monocular depth scaled to the points a view observes, and where
  those are all in the foreground the building behind comes out several times too near (a night
  view across the river lost its whole mask that way).

The same place is marked in views of either period: in a photo from before the fire the mask covers
the spire and the roofs, in one from after it the place where they were and the scaffolding.

What the masks do not know: foliage in front of the building (trees are never stable points, so a
tree that hides the choir is masked with it), cameras that COLMAP placed badly (the mask is then
as misplaced as the camera), and surfaces without texture: the lead roof of the nave has few
points away from its ridge and eaves, and in close views from the south-west, where pinnacles
stand around the lower roof, only part of it is masked.

Views of the west facade are a case of their own. The spire shows between the two towers in many of
them, but stable points of the towers and of the gallery surround it in the image, so the occluder
test takes it for hidden, as it rightly does with the roof. What such a view keeps is a few hundred
pixels of specks, some on the spire and others on the facade, and that is not a region to score.
Measured on 2026-11-20: of 12 frontal views looked at, 4 clearly show the spire and one of them has
part of it masked; of the 60 changed points that frontal views themselves observe, the occluder test
hides 85% (in views from the south, south-east and apse under 6%, and there 99-100% of the changed
points a view observes lie inside its mask). Of the 484 frontal views 57% have a mask of at least
256 pixels, 10% one of at least 2% of the frame; of the views from the south, south-east and apse
86-98% have the latter.

:func:`latent_weights` reduces a mask to the resolution of the latents by area averaging.
:func:`view_region_cells` does so for every view of the scene, in the form in which
``scripts/mvgen/prepare_regions.py`` stores the masks in the dataset folder (``changed_regions.npz``).
From there a scene hands them to its samples (``timemachine.mvgen.data``), for a loss that weights
the changed region and for the date-loss probe's numbers inside and outside it.

The module needs SciPy (nearest neighbours, connected groups, morphology), which MVGenMaster's
environment has.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from timemachine.data.dataset import crop_intrinsics
from timemachine.mvgen.data import LATENT_FACTOR, MVGenScene, ViewSamplingConfig

FIRE = ViewSamplingConfig.period_edges[-1]      # Notre-Dame's last period edge: the fire of 15 April 2019
KINDS = ("gone", "new")                         # observed only before the edge / only after it
RULES = ("candidates", "testable", "flagged", "clustered", "kept")      # what ChangedPoints.counts counts
SPARSE_ARRAYS = ("p3d_ids", "p3d_xyz", "view_starts", "view_pid")
CAMERA_COLUMNS = ("fx", "fy", "cx", "cy", "k1", "k2", "camera_width", "camera_height")
_DAYS_PER_YEAR = 365.25
_MASK_REFERENCE = 512                           # the resolution at which RegionConfig's pixel sizes are given


@dataclass(frozen=True)
class RegionConfig:
    """The thresholds of the changed-point rules and of the masks (see the module docstring).

    Pixel sizes are for 512 x 512 crops and scale with the resolution. No threshold is a length in
    COLMAP's arbitrary unit: lengths are multiples of the spacing of the changed points.

    What was measured on Notre-Dame (``data/datasets/25512_model0_v2``, 630 views before the fire and
    408 after it, 2026-11-20) to choose the defaults; "kept" is the number of points after rule 5,
    3,822 with the defaults (2,610 gone and 1,212 new):

    * ``min_days`` 3. 240 of the 408 views after the fire are from April and May 2019, up to 39 on one
      day. 2 days keep 5,354 points and 5 days 1,069; the masks of views from the south are then up
      to two fifths larger, and about half as large.
    * ``max_angle`` 30 and ``scale_margin`` 1.25. 20 degrees, or a margin of 1.5, keep nearly the same
      points (3,524 and 3,891) and the same masks; 45 degrees keep 3,101.
    * ``min_expected`` 3. Of 98,830 gone candidates 32,359 are testable and 10,102 flagged; of 43,147
      new candidates 29,540 and 6,749. Independent views would flag 5% of the testable points of an
      unchanged surface (exp(-3)); the 31% and 23% flagged show how far from independent they are,
      and why rule 4 is needed. With 5 the masks lose a tenth of their area. With 2 more points
      pass rule 4 (9,568 instead of 6,814), but they lie closer, rule 5's linking distance shrinks
      with them, part of the roofs falls out of the main group (3,577 kept) and some masks lose
      a quarter.
    * ``neighbours`` 24 and ``min_flagged_share`` 0.75. The share of flagged points among a flagged
      point's neighbours has two modes: at most 0.4 for 34% of the flagged gone points (chance) and
      at least 0.9 for 35% (change), with 21% between 0.4 and 0.75; for the new points 42%, 24% and
      25%. Rule 4 keeps 4,571 gone and 2,243 new points. A share of 0.6 keeps 5,114 points after
      rule 5 and doubles the mask of a view of the apse (from 13% of the frame to 26%); 12
      neighbours keep 4,469 (18% of that frame). A share of 0.9 keeps 3,261, 48 neighbours 3,586.
    * ``spacing_neighbour`` 4, ``link_factor`` 3 and ``min_group_share`` 0.2. The spacing is 0.021
      COLMAP units (about 0.9 m), the linking distance 0.063. The largest group holds 56% of the
      6,814 points and is the roofs, the spire and the crossing (its lowest point is two fifths
      up the towers); the next ones hold 6% or less: the choir's buttresses, things at the foot
      of the apse and of the west facade, the quays. Factors from 2.8 to 3.5 keep the same
      group; 2.5 keeps 3,108 points and 4 keeps 4,734, doubling the mask of a view of the apse.
      This is the most fragile threshold: look at the overlay sheets again
      (``scripts/mvgen/show_regions.py``) after a rebuild of the dataset or a change of the
      thresholds above.
    * ``splat_neighbour`` 2, ``min_pixels`` 2, ``max_pixels`` 16, ``closing_pixels`` 4, ``min_area`` 60:
      the smallest discs with which the roofs are covered without holes in views from the south.
      (Discs that reach the 4th neighbour, up to a whole linking distance and 24 pixels, with a
      closing of 6, give masks a fifth larger.)
    * ``occluders`` 64 within ``occluder_pixels`` 40, ``occluder_margin`` 2 linking distances (about
      5.5 m). Seven frontal views of the west facade keep 0.0-0.2% of the frame (a close-up 1.1%);
      without the from-behind test 0.0-0.7% (3.3%), without the occluder test 4-9% (25%), with
      neither 10-20% (34%). What did not work: a margin relative to the depth (15%) left what is a
      few metres behind a wall on that wall; 32 occluders within 24 pixels left 1% of the frame
      in frontal views and 6.5% in the close-up; and hiding a point when most of its 5 nearest
      occluders were in front of it, without asking that they surround it, took a little more
      off the roofs behind the pinnacles at their eaves (5.1% of the frame instead of 5.8% in a
      view from the south-west).
    """
    # --- changed points
    min_days: int = 3                   # rule 1: different days on which the point's own side observes it
    max_angle: float = 30.0             # rule 2: degrees between a view's direction and the observers' mean direction
    scale_margin: float = 1.25          # rule 2: the observers' range of focal length / distance is widened by this
    min_expected: float = 3.0           # rule 3: expected observers on the other side for a point to be testable
    neighbours: int = 24                # rule 4: nearest testable points looked at (the point itself included)
    min_flagged_share: float = 0.75     # rule 4: share of them that must be flagged
    spacing_neighbour: int = 4          # rule 5: the spacing is the median distance to a point's this-th nearest point
    link_factor: float = 3.0            # rule 5: points closer than this many spacings belong to one group
    min_group_share: float = 0.2        # rule 5: a group is kept if it holds this share of the points
    # --- masks
    splat_neighbour: int = 2            # a point's 3D radius: the distance to its this-th nearest changed point
    min_pixels: float = 2.0             # smallest and largest disc radius in the image
    max_pixels: float = 16.0
    occluders: int = 64                 # a point is hidden if, among its this-many nearest stable points in the image
    occluder_pixels: float = 40.0       # ... that are no farther from it than this,
    occluder_margin: float = 2.0        # ... some that are nearer by this many linking distances (rule 5's) surround it
    closing_pixels: int = 4             # gaps up to about twice this wide are closed
    min_area: int = 60                  # connected mask areas smaller than this many pixels are removed

    def __post_init__(self):
        if min(self.min_days, self.neighbours, self.spacing_neighbour, self.splat_neighbour) < 1:
            raise ValueError("min_days, neighbours, spacing_neighbour and splat_neighbour must be at least 1.")
        if not 0 < self.max_angle <= 180 or self.scale_margin < 1 or self.min_expected <= 0:
            raise ValueError("max_angle must be in (0, 180], scale_margin at least 1 and min_expected positive.")
        if not 0 < self.min_flagged_share <= 1 or not 0 <= self.min_group_share <= 1 or self.link_factor <= 0:
            raise ValueError("min_flagged_share must be in (0, 1], min_group_share in [0, 1] and link_factor positive.")
        if not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("disc radii need 0 < min_pixels <= max_pixels.")
        if self.occluders < 1 or self.occluder_pixels <= 0 or self.occluder_margin < 0:
            raise ValueError("occluders must be at least 1, occluder_pixels positive and occluder_margin not negative.")
        if self.closing_pixels < 0 or self.min_area < 0:
            raise ValueError("closing_pixels and min_area must not be negative.")


@dataclass(frozen=True)
class ChangedPoints:
    """The result of :func:`changed_points`.

    * ``gone``, ``new``: the kept points, ``[G, 3]`` and ``[M, 3]`` in COLMAP units;
    * ``gone_seen_from``, ``new_seen_from``: for each of them the mean of the unit vectors towards its
      observers. Its direction is the side of a building that the point is on; its length says how
      much the observers agree on that (1: all in one direction, near 0: all around the point);
    * ``stable``: ``[S, 3]`` the points that views of both sides observe. They were there throughout,
      and can hide a changed point from a view;
    * ``link``: the distance under which two changed points count as neighbours (rule 5's, in COLMAP
      units; 0 without points);
    * ``counts``: per kind, how many points each rule left: ``candidates`` (rule 1), ``testable`` and
      ``flagged`` (rule 3), ``clustered`` (rule 4) and ``kept`` (rule 5).
    """
    gone: np.ndarray
    new: np.ndarray
    gone_seen_from: np.ndarray
    new_seen_from: np.ndarray
    stable: np.ndarray
    link: float = 0.0
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def xyz(self) -> np.ndarray:
        """Both kinds together, ``[G + M, 3]``: the gone points first."""
        return np.concatenate([self.gone, self.new])

    @property
    def seen_from(self) -> np.ndarray:
        """The observers' mean directions of :attr:`xyz`, ``[G + M, 3]``."""
        return np.concatenate([self.gone_seen_from, self.new_seen_from])


# ----------------------------------------------------------------------------
# Changed points
# ----------------------------------------------------------------------------
def view_sides(dates: np.ndarray, usable: np.ndarray, edge: float, period_edges: Sequence[float] = ()) -> np.ndarray:
    """``[N]`` int: 0 for the usable views of the period that ends at ``edge``, 1 for those of the period
    that starts there, -1 for every other view.

    The two periods are bounded by the entries of ``period_edges`` next to ``edge`` (which need not be
    one of them); without such an entry a period is open-ended.
    """
    dates, usable = np.asarray(dates, dtype=np.float64), np.asarray(usable, dtype=bool)
    lower = max((float(e) for e in period_edges if e < edge), default=-np.inf)
    upper = min((float(e) for e in period_edges if e > edge), default=np.inf)
    side = np.full(len(dates), -1, dtype=np.int64)
    side[usable & (dates >= lower) & (dates < edge)] = 0
    side[usable & (dates >= edge) & (dates < upper)] = 1
    return side


def load_sparse(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    """The arrays of a dataset folder's ``sparse_points.npz``, read once."""
    with np.load(Path(dataset_dir) / "sparse_points.npz") as archive:
        return {name: archive[name] for name in SPARSE_ARRAYS}


def in_full_image(cam: np.ndarray, camera: Mapping[str, float]) -> np.ndarray:
    """``[P]`` bool: whether points in camera coordinates (``[P, 3]``) project inside the view's full image.

    ``camera`` holds the view's COLMAP camera (``CAMERA_COLUMNS``). COLMAP's observations are features
    of the full, distorted image, so its radial lens model is applied (``x_d = x (1 + k1 r^2 + k2 r^4)``),
    and only where the model is valid: where the distorted radius still grows with the true one.
    Beyond that the polynomial folds rays from far outside the field of view back into the image.
    """
    z = cam[:, 2]
    front = z > 0
    safe_z = np.where(front, z, 1.0)
    x, y = cam[:, 0] / safe_z, cam[:, 1] / safe_z
    r2 = x * x + y * y
    k1, k2 = camera["k1"], camera["k2"]
    stretch = 1 + k1 * r2 + k2 * r2 * r2
    growing = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2 > 0
    u, v = camera["fx"] * x * stretch + camera["cx"], camera["fy"] * y * stretch + camera["cy"]
    return front & growing & (u >= 0) & (u < camera["camera_width"]) & (v >= 0) & (v < camera["camera_height"])


def observation_pairs(sparse: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Every (view, point) observation once: view rows and indices into ``p3d_xyz``, sorted by view.

    A view can observe a point with two of its features; that is one observation here.
    """
    n_points = len(sparse["p3d_ids"])
    view = np.repeat(np.arange(len(sparse["view_starts"]) - 1), np.diff(sparse["view_starts"]))
    point = np.searchsorted(sparse["p3d_ids"], sparse["view_pid"]).clip(max=n_points - 1)
    if not np.array_equal(sparse["p3d_ids"][point], sparse["view_pid"]):
        raise ValueError("sparse points: a view observes point ids that are not in p3d_ids")
    pairs = np.unique(view.astype(np.int64) * n_points + point)
    return pairs // n_points, pairs % n_points


def widest_cosine(resultant: np.ndarray, max_angle: float) -> np.ndarray:
    """The smallest cosine between a view's direction and the mean direction of a point's observers at which
    the view still counts as looking at the point from the observers' side.

    ``resultant`` is the length of the mean of the unit vectors towards the observers. At 1 (all
    observers in one direction) the view may be ``max_angle`` degrees away from them. The mean
    direction of observers who stood all around a point says nothing, so the allowance widens with
    their spread, linearly in the cosine, to every direction at 0.
    """
    return resultant * np.cos(np.radians(max_angle)) - (1 - resultant)


def _flagged_points(xyz: np.ndarray, obs_view: np.ndarray, obs_point: np.ndarray, cameras: pd.DataFrame,
                    c2ws: np.ndarray, day: np.ndarray, side: np.ndarray, own: int, cfg: RegionConfig
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rules 1 to 3 for the points that side ``own`` observes: indices into ``xyz`` of the candidates, bool
    arrays over them saying which are testable and which are flagged, and ``[C, 3]`` the mean of the unit
    vectors from each candidate to its observers.

    ``obs_view`` / ``obs_point`` are the observations by the views of the two sides, sorted by view;
    ``day`` numbers the day of each view's date.
    """
    other = 1 - own
    mine = side[obs_view] == own
    n_other = np.bincount(obs_point[~mine], minlength=len(xyz))
    point_days = np.unique(obs_point[mine] * (day.max() + 1) + day[obs_view[mine]]) // (day.max() + 1)
    candidates = np.flatnonzero(np.bincount(point_days, minlength=len(xyz)) >= cfg.min_days)
    points = xyz[candidates]
    slot = np.full(len(xyz), -1, dtype=np.int64)
    slot[candidates] = np.arange(len(candidates))

    # The observers of each candidate on its own side: how finely they resolve it (focal length over
    # distance, the pixels one unit of length covers there) and the mean direction towards them.
    mine &= slot[obs_point] >= 0
    view, point = obs_view[mine], slot[obs_point[mine]]
    centres, focal = c2ws[:, :3, 3], cameras["fx"].to_numpy(dtype=np.float64)
    to_camera = centres[view] - points[point]
    distance = np.linalg.norm(to_camera, axis=1)
    coarsest, finest = np.full(len(points), np.inf), np.zeros(len(points))
    np.minimum.at(coarsest, point, focal[view] / distance)
    np.maximum.at(finest, point, focal[view] / distance)
    seen_from = np.stack([np.bincount(point, weights=to_camera[:, axis] / distance, minlength=len(points))
                          for axis in range(3)], axis=1) / np.bincount(point, minlength=len(points))[:, None]
    resultant = np.linalg.norm(seen_from, axis=1)
    direction = seen_from / np.maximum(resultant, 1e-12)[:, None]
    min_cos = widest_cosine(resultant, cfg.max_angle)

    # Rule 2, view by view: how many views of each side look at a point as its observers do, and how many
    # of those on its own side observe it.
    w2cs = np.linalg.inv(c2ws)
    looking = np.zeros((2, len(points)), dtype=np.int64)
    observing = np.zeros(len(points), dtype=np.int64)
    starts = np.searchsorted(view, np.arange(len(c2ws) + 1))        # the observations are sorted by view
    records = cameras[list(CAMERA_COLUMNS)].to_dict("records")
    for row in np.flatnonzero(side >= 0):
        to_camera = centres[row] - points
        distance = np.linalg.norm(to_camera, axis=1)
        scale = focal[row] / np.maximum(distance, 1e-12)
        alike = ((scale >= coarsest / cfg.scale_margin) & (scale <= finest * cfg.scale_margin)
                 & ((to_camera * direction).sum(axis=1) >= min_cos * distance))
        alike &= in_full_image(points @ w2cs[row, :3, :3].T + w2cs[row, :3, 3], records[row])
        looking[side[row]] += alike
        if side[row] == own:
            seen = point[starts[row]:starts[row + 1]]
            observing[seen] += alike[seen]

    expected = observing / np.maximum(looking[own], 1) * looking[other]
    testable = expected >= cfg.min_expected
    return candidates, testable, testable & (n_other[candidates] == 0), seen_from


def spacing(xyz: np.ndarray, neighbour: int) -> np.ndarray:
    """``[P]`` the distance from each point to its ``neighbour``-th nearest other point (0 for a lone point)."""
    if len(xyz) <= 1:
        return np.zeros(len(xyz))
    k = min(neighbour, len(xyz) - 1)
    return cKDTree(xyz).query(xyz, k=k + 1)[0][:, k]


def main_groups(xyz: np.ndarray, link: float, min_share: float) -> np.ndarray:
    """``[P]`` bool: the points in groups that hold at least ``min_share`` of all points.

    Two points are in one group if a chain of points leads from one to the other in steps of at
    most ``link`` (single linkage).
    """
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    pairs = cKDTree(xyz).query_pairs(link, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(xyz), len(xyz)))
    group = connected_components(graph, directed=False)[1]
    return (np.bincount(group) >= min_share * len(xyz))[group]


def changed_points(sparse: Mapping[str, np.ndarray], cameras: pd.DataFrame, c2ws: np.ndarray, dates: np.ndarray,
                   side: np.ndarray, cfg: RegionConfig = RegionConfig()) -> ChangedPoints:
    """The COLMAP points that one side of a period edge observes and the other should, but does not.

    Args:
        sparse: the arrays of ``sparse_points.npz`` (``p3d_ids`` sorted, ``p3d_xyz``, ``view_starts``,
            ``view_pid``).
        cameras: the view table, or any table with its COLMAP camera columns (``CAMERA_COLUMNS``), one
            row per view.
        c2ws: ``[N, 4, 4]`` camera-to-world of the views in COLMAP units (``MVGenScene.c2ws``).
        dates: ``[N]`` the views' dates in decimal years; two views are of one day if their dates are.
        side: ``[N]`` 0 for the views before the edge, 1 for those after it, -1 for views that are not
            used (:func:`view_sides`).

    Returns the gone points (seen only before) and the new points (seen only after) that pass the
    five rules of the module docstring, with the number of points each rule left, and the stable
    points (seen from both sides).
    """
    xyz = np.asarray(sparse["p3d_xyz"], dtype=np.float64)
    side = np.asarray(side, dtype=np.int64)
    day = np.unique(np.floor(np.asarray(dates, dtype=np.float64) * _DAYS_PER_YEAR), return_inverse=True)[1]
    obs_view, obs_point = observation_pairs(sparse)
    used = side[obs_view] >= 0
    obs_view, obs_point = obs_view[used], obs_point[used]
    clustered, seen_from, counts = [], [], {}
    for own, kind in enumerate(KINDS):
        candidates, testable, flagged, directions = _flagged_points(xyz, obs_view, obs_point, cameras, c2ws, day, side,
                                                                    own, cfg)
        chosen = np.flatnonzero(flagged)
        if len(chosen):
            # Rule 4: among a flagged point's nearest testable points, most must be flagged too.
            pool = np.flatnonzero(testable)
            nearest = cKDTree(xyz[candidates[pool]]).query(xyz[candidates[chosen]], k=min(cfg.neighbours, len(pool)))[1]
            share = flagged[pool][nearest.reshape(len(chosen), -1)].mean(axis=1)
            chosen = chosen[share >= cfg.min_flagged_share]
        clustered.append(xyz[candidates[chosen]])
        seen_from.append(directions[chosen])
        counts[kind] = {"candidates": len(candidates), "testable": int(testable.sum()),
                        "flagged": int(flagged.sum()), "clustered": len(chosen)}

    # Rule 5: the points of both kinds that belong to the main change.
    both = np.concatenate(clustered)
    link = cfg.link_factor * float(np.median(spacing(both, cfg.spacing_neighbour))) if len(both) else 0.0
    main = main_groups(both, link, cfg.min_group_share)
    main = {"gone": main[:len(clustered[0])], "new": main[len(clustered[0]):]}
    for kind in KINDS:
        counts[kind]["kept"] = int(main[kind].sum())
    observed = [np.bincount(obs_point[side[obs_view] == own], minlength=len(xyz)) > 0 for own in (0, 1)]
    return ChangedPoints(gone=clustered[0][main["gone"]], new=clustered[1][main["new"]],
                         gone_seen_from=seen_from[0][main["gone"]], new_seen_from=seen_from[1][main["new"]],
                         stable=xyz[observed[0] & observed[1]], link=link, counts=counts)


# ----------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------
def seen_from_behind(to_camera: np.ndarray, seen_from: np.ndarray) -> np.ndarray:
    """``[P]`` bool: the points that a camera looks at from the side their observers were not on.

    Args:
        to_camera: ``[P, 3]`` vectors from the points to the camera (any length).
        seen_from: ``[P, 3]`` the mean of the unit vectors from each point to its observers
            (``ChangedPoints.seen_from``), of length R <= 1.

    With observers in one direction (R = 1) the camera is behind the point when it is more than 90
    degrees away from that direction: it then sees the back of whatever the point is on. The more
    the observers were spread around the point, the farther round the camera may be
    (:func:`widest_cosine`), and a point that was seen from all sides, like the tip of a spire, has
    no back.
    """
    resultant = np.linalg.norm(seen_from, axis=1)
    cosine = (to_camera * seen_from).sum(axis=1) / np.maximum(np.linalg.norm(to_camera, axis=1) * resultant, 1e-12)
    return cosine < widest_cosine(resultant, 90.0)


def hidden_behind(uv: np.ndarray, z: np.ndarray, occluder_uv: np.ndarray, occluder_z: np.ndarray, count: int,
                  reach: float, margin: float) -> np.ndarray:
    """``[P]`` bool: the points of an image that occluders lie in front of.

    Args:
        uv, z: ``[P, 2]`` image positions and ``[P]`` depths of the points.
        occluder_uv, occluder_z: the same of the occluders (``[S, 2]``, ``[S]``).
        count, reach: the ``count`` occluders nearest to a point in the image are looked at, as far as
            they lie within ``reach`` pixels of it.
        margin: an occluder is in front of a point if it is nearer by more than this.

    A point is hidden if occluders in front of it surround it in the image: if there is one in each
    of the four quadrants around it (left or right of it, above or below). Behind a wall that is
    so; next to the outline of something nearer it is not: a roof seen above the pinnacles that
    stand in front of its eaves has them on one side only, and stays.
    """
    if len(uv) == 0 or len(occluder_uv) == 0:
        return np.zeros(len(uv), dtype=bool)
    nearest = cKDTree(occluder_uv).query(uv, k=count, distance_upper_bound=reach)[1].reshape(len(uv), -1)
    found = nearest < len(occluder_uv)                          # fewer than ``count`` may be in reach
    nearest = np.where(found, nearest, 0)
    in_front = found & (occluder_z[nearest] < z[:, None] - margin)
    offset = occluder_uv[nearest] - uv[:, None, :]
    quadrant = (offset[..., 0] > 0) + 2 * (offset[..., 1] > 0)
    return np.all([(in_front & (quadrant == q)).any(axis=1) for q in range(4)], axis=0)


def disc(radius: int) -> np.ndarray:
    """A filled circle of the given radius as a bool ``[2 radius + 1, 2 radius + 1]`` array."""
    offsets = np.arange(-radius, radius + 1)
    return offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius * radius


def latent_weights(mask: np.ndarray, factor: int = 8) -> np.ndarray:
    """A mask at the latents' resolution: float32 ``[H / factor, W / factor]`` in [0, 1], the masked share of
    the ``factor`` x ``factor`` pixels that each latent cell stands for (area averaging).

    Its sum times ``factor ** 2`` is the number of masked pixels.
    """
    height, width = mask.shape
    if height % factor or width % factor:
        raise ValueError(f"a {height} x {width} mask cannot be averaged over {factor} x {factor} cells")
    cells = np.asarray(mask, dtype=np.float32).reshape(height // factor, factor, width // factor, factor)
    return cells.mean(axis=(1, 3))


def view_region_cells(regions: ChangedRegions, resolution: int = 512, min_share: float = 0.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Every view's mask as a dataset folder stores it (``timemachine.mvgen.data.write_region_file``):
    ``(cells, share)``.

    ``cells`` is uint8 ``[N, resolution / 8, resolution / 8]``: per latent cell the number of masked
    pixels among the 8 x 8 it stands for, which is :func:`latent_weights` of the view's mask times 64,
    exactly. ``share`` is ``[N]``, the masked share of each view's frame.

    A mask that covers less than ``min_share`` of the frame is stored empty (its share is kept): in
    frontal views the mask is specks on unrelated stone, not the spire (the module docstring's
    paragraph on views of the west facade), and such a view must not get a region of its own. The
    tenth of the frontal views whose specks exceed 2% of the frame (close-ups, mostly) keep theirs
    at that threshold. A view without a readable image has no mask: its cells are empty and its
    share is NaN.
    """
    n_views, side = len(regions.scene.views), resolution // LATENT_FACTOR
    cells = np.zeros((n_views, side, side), dtype=np.uint8)
    share = np.full(n_views, np.nan)
    for row in range(n_views):
        try:
            mask = regions.mask(row, resolution)
        except (OSError, ValueError):       # no image file, or a lens model that crops are not made for
            continue
        share[row] = mask.mean()
        if share[row] >= min_share:
            cells[row] = np.rint(latent_weights(mask, LATENT_FACTOR) * LATENT_FACTOR ** 2)
    return cells, share


class ChangedRegions:
    """Masks of where a scene changed at a period edge, for any view of it (see the module docstring).

    Building it finds the changed points (about 12 seconds for Notre-Dame, on one CPU core); a mask is
    then made on first request (0.05 seconds) and kept, so asking again costs nothing.

    Attributes: ``points`` (:class:`ChangedPoints`, with the number of points each rule left), ``xyz``
    and ``seen_from`` (``[P, 3]`` the points of both kinds in COLMAP units, and the mean directions
    towards their observers), ``radius`` (``[P]`` the 3D radius each is drawn with) and ``side``
    (``[N]``: the views that stood for "before" (0) and "after" (1)).

    Args:
        scene: the scene; its ``config.period_edges`` bound the two periods that are compared.
        edge: the date (decimal year) at which the scene changed.
        config: the thresholds.
    """

    def __init__(self, scene: MVGenScene, edge: float = FIRE, config: RegionConfig = RegionConfig()):
        self.scene, self.edge, self.config = scene, float(edge), config
        self.side = view_sides(scene.dates, scene.usable, edge, scene.config.period_edges)
        self.points = changed_points(load_sparse(scene.dataset_dir), scene.views, scene.c2ws, scene.dates, self.side,
                                     config)
        self.xyz, self.seen_from = self.points.xyz, self.points.seen_from
        # A disc reaches the point's neighbours, so that points on one surface merge, but no farther than half
        # the linking distance: two points that are just close enough to be grouped then touch.
        self.radius = np.minimum(spacing(self.xyz, config.splat_neighbour), self.points.link / 2)
        self._masks: dict[tuple[int, int], np.ndarray] = {}        # (row, resolution) -> the mask's bits, packed

    def _project(self, xyz: np.ndarray, row: int, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop pixel coordinates ``[Q, 2]`` and depths ``[Q]`` of the points in front of view ``row``'s camera,
        and which of ``xyz`` those are (``[P]`` bool)."""
        w2c = np.linalg.inv(self.scene.c2ws[row])
        cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        front = cam[:, 2] > 1e-9
        cam = cam[front]
        uv = np.column_stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]])
        return uv, cam[:, 2], front

    def _draw(self, row: int, resolution: int) -> np.ndarray:
        """The mask of one view (see :meth:`mask`)."""
        cfg, pixels = self.config, resolution / _MASK_REFERENCE
        K = crop_intrinsics(next(self.scene.views.iloc[[row]].itertuples()), resolution)
        uv, z, front = self._project(self.xyz, row, K)
        size = np.clip(K[0, 0] * self.radius[front] / z, cfg.min_pixels * pixels, cfg.max_pixels * pixels)
        shown = ((uv > -size[:, None]) & (uv < resolution + size[:, None])).all(axis=1)     # the disc meets the frame
        shown &= ~seen_from_behind(self.scene.c2ws[row, :3, 3] - self.xyz[front], self.seen_from[front])
        uv, z, size = uv[shown], z[shown], size[shown]
        stable_uv, stable_z, _ = self._project(self.points.stable, row, K)
        visible = ~hidden_behind(uv, z, stable_uv, stable_z, cfg.occluders, cfg.occluder_pixels * pixels,
                                 cfg.occluder_margin * self.points.link)

        canvas = Image.new("1", (resolution, resolution), 0)
        draw = ImageDraw.Draw(canvas)
        for (x, y), r in zip(uv[visible] - 0.5, size[visible]):                 # PIL's pixel centres are integers
            draw.ellipse((x - r, y - r, x + r, y + r), fill=1)
        mask = np.asarray(canvas, dtype=bool)

        closing = round(cfg.closing_pixels * pixels)
        if closing > 0:
            # Dilate, then erode. The erosion treats the outside of the frame as masked, so that an area
            # which runs out of the frame is not eaten back from the border.
            mask = ndimage.binary_erosion(ndimage.binary_dilation(mask, disc(closing)), disc(closing), border_value=1)
        areas, n_areas = ndimage.label(mask)
        small = np.bincount(areas.ravel(), minlength=n_areas + 1) < cfg.min_area * pixels * pixels
        small[0] = False                                                        # label 0 is the background
        return mask & ~small[areas]

    def mask(self, row: int, resolution: int = 512) -> np.ndarray:
        """Bool ``[resolution, resolution]``: where view ``row`` shows (or showed, or would show) what changed
        at the edge, aligned with ``scene.make_sample``'s image of that view at that resolution.

        Works for any row of the view table that has an image file, usable or not. The array is the
        caller's own: changing it does not change what the next call returns.
        """
        key = (int(row), int(resolution))
        if key not in self._masks:
            self._masks[key] = np.packbits(self._draw(*key))
        bits = np.unpackbits(self._masks[key], count=resolution * resolution)
        return bits.reshape(resolution, resolution).astype(bool)

    def share(self, row: int, resolution: int = 512) -> float:
        """The masked fraction of view ``row``'s frame."""
        return float(self.mask(row, resolution).mean())
