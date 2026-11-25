"""The per-view look code: 24 fixed colour statistics and the photo's medium, and how training uses them.

Experiment look1 (2026-11-24) asks whether an explicit, untrained description of a photo's look lets
the date learn the state of the place instead of the look of the photos taken at that time. The
look is given to the model per view, for references AND targets, so that the look of an output can
be chosen. This module holds everything about the code that is not a network: its arithmetic, the
files it is stored in, its standardisation, the references' code that inference uses by default,
the medium conversion of arm D, and the random draws a training sample gets. The embedding that
feeds the code to the UNet is ``timemachine.mvgen.date.LookEmbedding``.

**The code** (:func:`look_code`, ``LOOK_FIELDS``), from the 512 x 512 crop exactly as
``MVGenScene.make_sample`` gives it, over the pixels outside the view's changed region (the stored
mask grown by one latent cell, 8 px at 512: :func:`excluded_pixels`) when the dataset folder has a
non-empty mask for the view, else over the whole frame:

* global (6): the mean and the log standard deviation (``log(sd + 1e-3)``) of each sRGB channel;
* class colour (15): the mean sRGB colour of sky, building, vegetation, ground and water
  (``LOOK_CLASSES``; groups of SegFormer-B2's ADE20K classes, ``ADE20K_GROUPS``, from class maps
  cached at 128 px). A class is *shown* when it covers at least 1% of the frame outside the region;
  the entries of a class the view does not show are NaN;
* tone (3): one-hot grey / toned / colour (``TONES``). For Notre-Dame the appearance study's
  eye-corrected labels are used (``scripts/mvgen/prepare_look.py --tone-labels``); elsewhere the
  residual-chroma rule (:func:`tone_rule`).

There are no presence bits and no night flag: the look1 critiques measured that presence bits mostly
repeat the camera and that a pixel night rule is about 85% precise, while the colour statistics carry
darkness anyway. The arithmetic is the appearance study's (``data/appearance_study/design_minimal``),
which computed it from 256 px crops; at 512 px the means agree within 0.01 sd and the log spreads are
about 0.08 sd higher (``data/look1_critique/feasibility/02b_code_at_512.log``).

**Inputs to the model** are standardised (:class:`LookStats`): each entry minus the mean of the
training views, over their standard deviation; an entry of an unshown class (NaN) becomes the
population mean, 0 after standardisation. In training (:class:`TrainingLooks`):

* an entry of a class the view does not show is first filled with that class's entry of a random
  other training view that shows it, so that an entry never says whether the class is in the frame;
* then each class entry is replaced, independently with probability ``look_class_unspecified`` (0.3),
  by the population mean ("unspecified");
* when a sample's dates are all dropped (date dropout), its looks are dropped too with probability
  ``look_dropout`` (0.5): the model then gets its learned "unknown look". The look is never dropped
  while the date is shown;
* arm D (``medium_grey``, ``medium_toned``): each COLOUR view, reference or target, independently
  becomes grey (luminance in all channels) or toned (the luminance mapped through the tone curve of a
  real toned print of the dataset, :func:`toned_curve`) at its unchanged date; its code is computed
  from the converted pixels and its tone set by construction.

Every one of these draws comes from generators of their own, seeded by ``(seed, epoch, sample index,
LOOK_STREAM, tag)`` as ``MVGenScene.draw`` seeds the view sets, one tag per kind of draw, each of a
fixed shape: a run with the look draws the same view sets, timesteps, noise and dropouts as the same
run without it, and arm D draws the same class fills and dropouts as arm B.

**At inference** a target gets the references' code by default (:func:`references_code`): the global
numbers and the tone from the mean and the majority over the references, a class entry from the mean
over the references that show the class, else the population mean.

**Files** in the dataset folder (``scripts/mvgen/prepare_look.py`` writes both):

* ``class_maps.npz`` (``CLASS_MAP_FILE``): ``labels`` uint8 ``[N, 128, 128]``, SegFormer-B2's ADE20K
  class of each view's 512 crop at 128 px; ``file_keys``, ``model``, ``revision``. A cache: the look
  code reads its class groups (:func:`class_groups`);
* ``look_codes.npz`` (``LOOK_FILE``): ``codes`` float32 ``[N, 24]`` (raw, NaN for unshown classes; a
  row of NaN for a view without an image), ``shown`` bool ``[N, 5]``, ``tone`` ``[N]`` (one of ``TONES``,
  ``""`` without an image), ``tone_source`` ``[N]`` (``"labels"`` or ``"rule"``), ``computed`` bool
  ``[N]``, ``region`` bool ``[N, 64, 64]`` (the latent cells left out of each code: the grown mask),
  ``toned_curves`` float32 ``[K, 256, 3]`` and ``toned_rows`` ``[K]`` (the tone curves of the views
  whose tone is "toned", for arm D), ``stats_mean`` / ``stats_sd`` (:class:`LookStats` of the usable
  views), ``file_keys`` and ``config`` (JSON: how the file was made). :func:`read_look` reads it.

Nothing here imports MVGenMaster's code, so the module also loads in our own environment; only
:func:`tone_rule` needs OpenCV (MVGenMaster's environment has it).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

LOOK_FILE = "look_codes.npz"
CLASS_MAP_FILE = "class_maps.npz"
LOOK_CLASSES = ("sky", "building", "vegetation", "ground", "water")
TONES = ("grey", "toned", "colour")
LOOK_FIELDS = (("mean_r", "mean_g", "mean_b", "log_sd_r", "log_sd_g", "log_sd_b")
               + tuple(f"{name}_{channel}" for name in LOOK_CLASSES for channel in "rgb")
               + tuple(f"tone_{tone}" for tone in TONES))
LOOK_DIM = len(LOOK_FIELDS)                                     # 24
GLOBAL = slice(0, 6)                                            # where the parts of a code are
CLASS_START = 6
TONE = slice(CLASS_START + 3 * len(LOOK_CLASSES), LOOK_DIM)

# The ADE20K classes of each look class (the appearance study's groups; SegFormer's label ids).
ADE20K_GROUPS = {"sky": (2,), "building": (1, 0, 25, 48, 84, 61), "vegetation": (4, 9, 17, 66, 72),
                 "ground": (6, 11, 13, 3, 52, 46, 53), "water": (21, 26, 60, 128)}
NO_CLASS = 255                                                  # a pixel of none of the five classes
SEGFORMER = "nvidia/segformer-b2-finetuned-ade-512-512"
SEGFORMER_REVISION = "de01bae28967510f9ddd496c60a969357195400c"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASS_MAP_SIZE = 128                                            # side of the cached class maps
MIN_CLASS_SHARE = 0.01                                          # share of the frame from which a class is shown
LOG_SD_EPS = 1e-3
REGION_GROW = 1                                                 # latent cells the stored mask is grown by (8 px)
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)        # Rec. 601, as the appearance study
TONE_CURVE_BINS, TONE_CURVE_MIN_PIXELS = 16, 20                 # luminance bins of a tone curve, pixels per bin
LOOK_STREAM = 7_100                                             # marks the training look draws' generators
LOOK_DRAWS = {"medium": 1, "fill": 2, "unspecified": 3, "dropout": 4}       # one generator per kind of draw
TONE_SOURCES = ("labels", "rule")

_LUT = np.full(256, NO_CLASS, dtype=np.uint8)
for _index, _ids in enumerate(ADE20K_GROUPS.values()):
    _LUT[list(_ids)] = _index


def class_slice(index: int) -> slice:
    """Where the three entries of look class ``index`` (into ``LOOK_CLASSES``) are in a code."""
    return slice(CLASS_START + 3 * index, CLASS_START + 3 * index + 3)


def class_groups(labels: np.ndarray) -> np.ndarray:
    """ADE20K label map (uint8, any shape) -> look-class map: the index in ``LOOK_CLASSES``, or ``NO_CLASS``."""
    return _LUT[np.asarray(labels, dtype=np.uint8)]


def _upsample(cells: np.ndarray, size: int) -> np.ndarray:
    """``[h, w]`` -> ``[size, size]`` by repeating each cell (``size`` a multiple of ``h`` and ``w``)."""
    h, w = cells.shape
    if size % h or size % w:
        raise ValueError(f"a map of {h} x {w} cells does not divide a frame of {size} pixels")
    return np.repeat(np.repeat(cells, size // h, axis=0), size // w, axis=1)


def grown_cells(cells: np.ndarray, grow: int = REGION_GROW) -> np.ndarray:
    """A bool cell mask ``[h, w]`` grown by ``grow`` cells in each direction (diagonals included)."""
    cells = np.asarray(cells, dtype=bool)
    h, w = cells.shape
    padded = np.pad(cells, grow)
    out = np.zeros_like(cells)
    for dy in range(2 * grow + 1):
        for dx in range(2 * grow + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def excluded_pixels(region_cells: np.ndarray, size: int) -> np.ndarray | None:
    """The pixels a code leaves out, bool ``[size, size]``, from the cells left out (``[h, w]``; already grown,
    as ``look_codes.npz`` stores them); None when no cell is left out (the code is then of the whole frame)."""
    region_cells = np.asarray(region_cells, dtype=bool)
    return _upsample(region_cells, size) if region_cells.any() else None


def tone_rule(image: np.ndarray) -> str:
    """The residual-chroma rule of the appearance study (``confounds/measures.py``): one of ``TONES``.

    The image (float ``[H, W, 3]`` in [0, 1]) is taken to 8 bits and 128 x 128 (area average) and to
    CIELAB. ``res_ab`` is what is left of the chroma (a, b) once it is predicted from the lightness by a
    cubic: a print's colour is a function of its lightness, a colour photo's is not. Below 0.3 the file
    is grey; below 1.7, or below 3.5 with cream highlights (mean b of the lightest fifth at least 9),
    toned; colour otherwise. The study's night exception is not used (there is no night flag). Needs
    OpenCV (MVGenMaster's environment).
    """
    try:
        import cv2
    except ImportError as error:                                # our own environment has no OpenCV
        raise ImportError("tone_rule needs OpenCV (cv2), which MVGenMaster's environment has") from error
    pixels = (np.clip(np.asarray(image, dtype=np.float32), 0, 1) * 255).round().astype(np.uint8)
    small = cv2.resize(pixels, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    light, ab = lab[:, 0] / 100.0, lab[:, 1:]
    design = np.stack([np.ones_like(light), light, light ** 2, light ** 3], 1)
    coefficients, *_ = np.linalg.lstsq(design, ab, rcond=None)
    res_ab = float(np.sqrt(((ab - design @ coefficients) ** 2).sum(1).mean()))
    b_hi = float(ab[light >= np.percentile(light, 80), 1].mean())
    if res_ab < 0.3:
        return "grey"
    if res_ab < 1.7 or (res_ab < 3.5 and b_hi >= 9.0):
        return "toned"
    return "colour"


def look_code(image: np.ndarray, class_map: np.ndarray, region: np.ndarray | None = None, tone: str | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """The raw look code of one view: ``(code float32 [24], shown bool [5])``.

    Args:
        image: float32 ``[H, W, 3]`` in [0, 1], the 512 crop of ``MVGenScene.make_sample``.
        class_map: uint8 ``[h, w]``, the look class of each cell (index in ``LOOK_CLASSES`` or ``NO_CLASS``;
            :func:`class_groups` of the cached labels); repeated to the frame.
        region: bool ``[H, W]``, the pixels to leave out (:func:`excluded_pixels`), or None for the whole frame.
            A region that leaves no pixel gives the code of the whole frame.
        tone: one of ``TONES``; None applies :func:`tone_rule` to the image.

    The entries of a class that covers less than ``MIN_CLASS_SHARE`` of the frame (outside the region)
    are NaN and the class is not shown.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"an image is [H, W, 3]; got {image.shape}")
    height, width = image.shape[:2]
    if height != width:
        raise ValueError(f"the look code is made from square crops; got {height} x {width}")
    groups = _upsample(np.asarray(class_map, dtype=np.uint8), height)
    keep = np.ones((height, width), dtype=bool) if region is None else ~np.asarray(region, dtype=bool)
    if not keep.any():
        keep = np.ones_like(keep)
    pixels = image[keep]
    code = np.full(LOOK_DIM, np.nan, dtype=np.float32)
    code[0:3] = pixels.mean(0)
    code[3:6] = np.log(pixels.std(0) + LOG_SD_EPS)
    shown = np.zeros(len(LOOK_CLASSES), dtype=bool)
    for index in range(len(LOOK_CLASSES)):
        selected = (groups == index) & keep
        if selected.sum() >= MIN_CLASS_SHARE * height * width:
            shown[index] = True
            code[class_slice(index)] = image[selected].mean(0)
    tone = tone_rule(image) if tone is None else tone
    if tone not in TONES:
        raise ValueError(f"tone must be one of {TONES}, got {tone!r}")
    code[TONE] = np.eye(len(TONES), dtype=np.float32)[TONES.index(tone)]
    return code, shown


def to_image(images: torch.Tensor) -> np.ndarray:
    """A sample's view ``[3, S, S]`` in [-1, 1] (``MVGenScene.make_sample``) as float32 ``[S, S, 3]`` in [0, 1]."""
    return ((images.detach().float().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0).astype(np.float32)


def from_image(image: np.ndarray) -> torch.Tensor:
    """Inverse of :func:`to_image`: float ``[S, S, 3]`` in [0, 1] -> ``[3, S, S]`` float32 in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.float32) * 2.0 - 1.0)).permute(2, 0, 1)


# ----------------------------------------------------------------------------
# Medium
# ----------------------------------------------------------------------------
def toned_curve(image: np.ndarray) -> np.ndarray:
    """The colour of a toned print as a function of its luminance: float32 ``[256, 3]``.

    The image's luminance (Rec. 601) is cut into ``TONE_CURVE_BINS`` bins; each bin with at least
    ``TONE_CURVE_MIN_PIXELS`` pixels gives its mean colour at the bin's centre, and the curve is the
    linear interpolation of those points at 256 luminance levels (held flat beyond the first and last).
    """
    image = np.asarray(image, dtype=np.float32)
    light = image @ LUMA
    bins = np.clip((light * TONE_CURVE_BINS).astype(int), 0, TONE_CURVE_BINS - 1)
    centres, colours = [], []
    for b in range(TONE_CURVE_BINS):
        inside = bins == b
        if inside.sum() >= TONE_CURVE_MIN_PIXELS:
            centres.append((b + 0.5) / TONE_CURVE_BINS)
            colours.append(image[inside].mean(0))
    if not centres:
        raise ValueError("the image has no luminance bin with enough pixels for a tone curve")
    colours = np.asarray(colours)
    levels = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(levels, centres, colours[:, c]) for c in range(3)], 1).astype(np.float32)


def convert_medium(image: np.ndarray, kind: str, curve: np.ndarray | None = None) -> np.ndarray:
    """A colour photo as a grey or a toned print: float32 ``[H, W, 3]`` in [0, 1].

    ``kind`` "grey": the luminance (Rec. 601) in all three channels. "toned": the luminance mapped through
    ``curve`` (``[256, 3]``, :func:`toned_curve` of a real toned print), at the nearest of its 256 levels.
    """
    light = np.clip(np.asarray(image, dtype=np.float32) @ LUMA, 0.0, 1.0)
    if kind == "grey":
        return np.repeat(light[..., None], 3, axis=-1)
    if kind == "toned":
        if curve is None:
            raise ValueError("a toned conversion needs a tone curve")
        return np.asarray(curve, dtype=np.float32)[np.clip(np.round(light * 255).astype(int), 0, 255)]
    raise ValueError(f"a medium conversion is 'grey' or 'toned', not {kind!r}")


# ----------------------------------------------------------------------------
# Standardisation and the references' code
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LookStats:
    """Per-entry mean and standard deviation of the raw codes of the training views.

    A class entry's mean is taken over the views that show the class: it is the population mean that
    an unshown or unspecified entry stands for. An entry without spread (or without any value) gets a
    standard deviation of 1 (and, without any value, a mean of 0).
    """

    mean: tuple[float, ...]
    sd: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "mean", tuple(float(x) for x in self.mean))
        object.__setattr__(self, "sd", tuple(float(x) for x in self.sd))
        if len(self.mean) != LOOK_DIM or len(self.sd) != LOOK_DIM:
            raise ValueError(f"look statistics need {LOOK_DIM} means and standard deviations")
        if not all(np.isfinite(self.mean)) or not all(s > 0 for s in self.sd):
            raise ValueError("look statistics need finite means and positive standard deviations")

    @classmethod
    def from_codes(cls, codes: np.ndarray) -> LookStats:
        """The statistics of raw codes ``[n, 24]`` (NaN entries are left out of their entry's numbers)."""
        codes = np.asarray(codes, dtype=np.float64)
        counts = np.isfinite(codes).sum(0)
        filled = np.where(np.isfinite(codes), codes, 0.0)
        mean = np.divide(filled.sum(0), counts, out=np.zeros(LOOK_DIM), where=counts > 0)
        squares = np.where(np.isfinite(codes), (codes - mean) ** 2, 0.0).sum(0)
        sd = np.sqrt(np.divide(squares, counts, out=np.zeros(LOOK_DIM), where=counts > 0))
        return cls(mean=tuple(mean), sd=tuple(np.where(sd > 0, sd, 1.0)))

    def standardise(self, codes: np.ndarray | torch.Tensor) -> np.ndarray:
        """Raw codes ``[..., 24]`` -> float32 standardised codes; NaN entries become 0 (the population mean)."""
        codes = codes.detach().cpu().numpy() if torch.is_tensor(codes) else np.asarray(codes)
        z = (codes.astype(np.float64) - np.asarray(self.mean)) / np.asarray(self.sd)
        return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": list(self.mean), "sd": list(self.sd)}

    @classmethod
    def from_dict(cls, stored: Mapping) -> LookStats:
        return cls(mean=tuple(stored["mean"]), sd=tuple(stored["sd"]))


def references_code(codes: np.ndarray, shown: np.ndarray) -> np.ndarray:
    """The code a target gets by default from its references: raw float32 ``[24]``.

    ``codes`` ``[n, 24]`` and ``shown`` ``[n, 5]`` are the references'. The global numbers are their
    mean; the tone is the one most of them have (a tie goes to the tied tone of the earliest
    reference); a class entry is the mean over the references that show the class, NaN (the population
    mean, once standardised) when none does.
    """
    codes, shown = np.asarray(codes, dtype=np.float32), np.asarray(shown, dtype=bool)
    if (codes.ndim != 2 or codes.shape[1] != LOOK_DIM or len(codes) == 0
            or shown.shape != (len(codes), len(LOOK_CLASSES))):
        raise ValueError(f"references_code needs codes [n, {LOOK_DIM}] and shown [n, {len(LOOK_CLASSES)}], n >= 1")
    out = np.full(LOOK_DIM, np.nan, dtype=np.float32)
    out[GLOBAL] = codes[:, GLOBAL].mean(0)
    for index in range(len(LOOK_CLASSES)):
        if shown[:, index].any():
            out[class_slice(index)] = codes[shown[:, index], class_slice(index)].mean(0)
    tones = codes[:, TONE].argmax(1)
    counts = np.bincount(tones, minlength=len(TONES))
    tied = np.flatnonzero(counts == counts.max())
    chosen = next(t for t in tones if t in tied)
    out[TONE] = np.eye(len(TONES), dtype=np.float32)[chosen]
    return out


# ----------------------------------------------------------------------------
# Donors: another view's look, matched on what the code allows
# ----------------------------------------------------------------------------
def season_of(date_lo: np.ndarray, date_hi: np.ndarray, precision: float = 0.09) -> np.ndarray:
    """The time of year of each view's date (0 to 1, from 1 January), NaN where the date is known to less
    than ``precision`` years (about a month)."""
    date_lo, date_hi = np.asarray(date_lo, dtype=np.float64), np.asarray(date_hi, dtype=np.float64)
    middle = (date_lo + date_hi) / 2
    return np.where(date_hi - date_lo <= precision, middle - np.floor(middle), np.nan)


def matched_donors(targets: Sequence[int], candidates: np.ndarray, look: Mapping[str, np.ndarray],
                   season: np.ndarray, stats: LookStats) -> np.ndarray:
    """For each target row, the candidate whose look is the closest match on tone, sky and season.

    Args:
        targets: rows of the view table.
        candidates: bool ``[len(targets), N]``, the rows each target may take a donor from (or ``[N]``, the same
            rows for every target).
        look: :func:`read_look` of the scene (``codes``, ``shown``, ``tone``).
        season: ``[N]`` :func:`season_of` the views' dates.
        stats: to measure sky colours in standard deviations.

    A donor of the target's tone is preferred to any other, then one that shows the sky if and only if
    the target does; among those the nearest by season (months / 3, where both seasons are known) plus
    sky colour (the distance of the two sky means in standard deviations, where both show the sky). A
    tie goes to the lowest row. ``-1`` for a target without any candidate.
    """
    candidates = np.asarray(candidates, dtype=bool)
    if candidates.ndim == 1:
        candidates = np.broadcast_to(candidates, (len(targets), len(candidates)))
    codes, shown, tone = look["codes"], np.asarray(look["shown"], dtype=bool), np.asarray(look["tone"])
    sky = class_slice(LOOK_CLASSES.index("sky"))
    sky_sd = np.asarray(stats.sd)[sky]
    has_sky = shown[:, LOOK_CLASSES.index("sky")]
    donors = np.full(len(targets), -1, dtype=np.int64)
    for i, target in enumerate(targets):
        rows = np.flatnonzero(candidates[i])
        if len(rows) == 0:
            continue
        months = np.abs(season[rows] - season[target])
        months = np.minimum(months, 1 - months) * 12 / 3
        cost = np.where(np.isfinite(months), months, 0.0)
        both_sky = has_sky[rows] & has_sky[target]
        sky_distance = np.sqrt((((codes[rows][:, sky] - codes[target, sky]) / sky_sd) ** 2).sum(1))
        cost = cost + np.where(both_sky, np.nan_to_num(sky_distance, nan=0.0), 0.0)
        cost = cost + 100.0 * (has_sky[rows] != has_sky[target]) + 1000.0 * (tone[rows] != tone[target])
        donors[i] = rows[np.argmin(cost)]                       # argmin takes the first, i.e. the lowest row
    return donors


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def _note(path: Path, problem: str) -> None:
    print(f"[scene] note: {path} is not used: {problem}. Run scripts/mvgen/prepare_look.py --overwrite to make it "
          "for this scene.", flush=True)


def write_look_file(dataset_dir: Path, arrays: Mapping[str, np.ndarray], config: Mapping[str, object]) -> Path:
    """Write ``look_codes.npz`` (layout in the module docstring) under another name, then rename it."""
    path = Path(dataset_dir) / LOOK_FILE
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as file:
        np.savez_compressed(file, **arrays, config=json.dumps(dict(config)))
    tmp.replace(path)
    return path


def read_look(dataset_dir: str | Path, file_keys: Sequence[str] | None = None) -> dict | None:
    """A dataset folder's look codes (``look_codes.npz``), or None when the folder has none.

    Returns ``codes`` float32 ``[N, 24]`` (raw, NaN for unshown classes), ``shown`` bool ``[N, 5]``, ``tone``
    ``[N]`` str, ``tone_source`` ``[N]`` str, ``computed`` bool ``[N]``, ``region`` bool ``[N, h, w]``,
    ``toned_curves`` ``[K, 256, 3]``, ``toned_rows`` ``[K]``, ``stats`` (:class:`LookStats` of the usable
    views), ``file_keys`` and ``config`` (dict). With ``file_keys`` (the view table's), a file made for
    another table is not used: a note is printed and None returned, as for a file that cannot be read.
    """
    path = Path(dataset_dir) / LOOK_FILE
    if not path.exists():
        return None
    try:
        with np.load(path) as stored:
            look = {name: stored[name] for name in stored.files}
    except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile, zlib.error) as error:
        _note(path, f"it cannot be read ({type(error).__name__}: {error})")
        return None
    if file_keys is not None and not np.array_equal(look["file_keys"], np.asarray(file_keys, dtype=str)):
        _note(path, "it was made for another view table")
        return None
    look["config"] = json.loads(str(look["config"]))
    look["stats"] = LookStats(mean=tuple(look.pop("stats_mean")), sd=tuple(look.pop("stats_sd")))
    return look


def read_class_maps(dataset_dir: str | Path, file_keys: Sequence[str] | None = None) -> np.ndarray | None:
    """The look-class maps of a dataset folder's ``class_maps.npz``: uint8 ``[N, 128, 128]`` (:func:`class_groups`
    of the stored ADE20K labels), or None without the file or, with ``file_keys``, for another view table."""
    path = Path(dataset_dir) / CLASS_MAP_FILE
    if not path.exists():
        return None
    with np.load(path) as stored:
        if file_keys is not None and not np.array_equal(stored["file_keys"], np.asarray(file_keys, dtype=str)):
            return None
        return class_groups(stored["labels"])


def look_fingerprint(look: Mapping[str, object]) -> str:
    """A hash (SHA-256, in hex) of what in a :func:`read_look` result decides a training sample's look input."""
    digest = hashlib.sha256()
    for name in ("codes", "shown", "computed", "region", "toned_curves", "toned_rows"):
        digest.update(np.ascontiguousarray(look[name]).tobytes())
    digest.update("\0".join(np.asarray(look["tone"], dtype=str)).encode("utf-8"))
    digest.update(json.dumps(look["config"], sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


# ----------------------------------------------------------------------------
# Training draws
# ----------------------------------------------------------------------------
class TrainingLooks:
    """The look input of training samples: class fill, unspecified entries, look dropout and arm D's medium.

    Called on a sample of ``MVGenScene.make_sample`` (which carries ``look_raw``, ``look_shown`` and ``tone``)
    with its epoch and index, it returns a copy with ``look`` (float32 ``[F, 24]``, standardised, what the
    UNet is given) and ``look_drop`` (bool: drop the sample's looks if its dates are dropped; the training
    step decides, ``timemachine.mvgen.step.training_loss``). With a medium conversion the converted views'
    ``images``, ``look_raw`` and ``tone`` are those of the converted pixels. The sample given is not changed.

    Args:
        look: :func:`read_look` of the scene.
        training: bool ``[N]``, the views that train (donors of class entries and of tone curves).
        stats: the run's :class:`LookStats` (of the training views).
        class_maps: :func:`read_class_maps` of the scene; needed only with a medium conversion.
        seed: the run's seed; with the epoch, the index and ``LOOK_STREAM`` it seeds the draws.
        dropout, unspecified, grey, toned: ``LossConfig.look_dropout``, ``look_class_unspecified``,
            ``medium_grey`` and ``medium_toned``.
    """

    def __init__(self, look: Mapping[str, np.ndarray], training: np.ndarray, stats: LookStats, *, seed: int,
                 dropout: float, unspecified: float, grey: float = 0.0, toned: float = 0.0,
                 class_maps: np.ndarray | None = None):
        self.stats, self.seed = stats, int(seed)
        self.dropout, self.unspecified, self.grey, self.toned = dropout, unspecified, grey, toned
        training = np.asarray(training, dtype=bool) & np.asarray(look["computed"], dtype=bool)
        shown = np.asarray(look["shown"], dtype=bool)
        # per class: the class entries of the training views that show it (the donors of a fill)
        self.donors = [np.asarray(look["codes"])[training & shown[:, c], class_slice(c)]
                       for c in range(len(LOOK_CLASSES))]
        self.converts = grey + toned > 0
        if self.converts:
            if class_maps is None:
                raise ValueError("a medium conversion recomputes codes and needs the scene's class maps "
                                 f"({CLASS_MAP_FILE})")
            keep = training[np.asarray(look["toned_rows"], dtype=np.int64)]
            self.curves = np.asarray(look["toned_curves"])[keep]
            if toned > 0 and len(self.curves) == 0:
                raise ValueError("a toned conversion needs the tone curve of at least one toned print among the "
                                 "training views")
            self.class_maps, self.region = class_maps, np.asarray(look["region"], dtype=bool)

    def _rng(self, epoch: int, index: int, draw: str) -> np.random.Generator:
        return np.random.default_rng((self.seed, int(epoch), int(index), LOOK_STREAM, LOOK_DRAWS[draw]))

    def __call__(self, sample: dict, epoch: int, index: int) -> dict:
        sample = dict(sample)
        raw = sample["look_raw"].numpy().astype(np.float32).copy()
        shown = sample["look_shown"].numpy().astype(bool).copy()
        tone = list(sample["tone"])
        n_views = len(raw)
        if self.converts:
            u, pick = self._rng(epoch, index, "medium").random((2, n_views))
            images = sample["images"].clone()
            for view in range(n_views):
                kind = "grey" if u[view] < self.grey else "toned" if u[view] < self.grey + self.toned else None
                if kind is None or tone[view] != "colour":
                    continue
                row = int(sample["rows"][view])
                curve = self.curves[int(pick[view] * len(self.curves))] if kind == "toned" else None
                converted = convert_medium(to_image(images[view]), kind, curve)
                raw[view], shown[view] = look_code(converted, self.class_maps[row],
                                                   excluded_pixels(self.region[row], converted.shape[0]), kind)
                images[view], tone[view] = from_image(converted), kind
            sample.update(images=images, look_raw=torch.from_numpy(raw.copy()), look_shown=torch.from_numpy(shown),
                          tone=tone)
        fill = self._rng(epoch, index, "fill").random((n_views, len(LOOK_CLASSES)))
        for view in range(n_views):
            for c in np.flatnonzero(~shown[view]):
                if len(self.donors[c]):
                    raw[view, class_slice(c)] = self.donors[c][int(fill[view, c] * len(self.donors[c]))]
        look = self.stats.standardise(raw)
        unspecified = self._rng(epoch, index, "unspecified").random((n_views, len(LOOK_CLASSES))) < self.unspecified
        for c in range(len(LOOK_CLASSES)):
            look[unspecified[:, c], class_slice(c)] = 0.0
        sample["look"] = torch.from_numpy(look)
        sample["look_drop"] = bool(self._rng(epoch, index, "dropout").random() < self.dropout)
        return sample
