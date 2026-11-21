"""How much of our "era" signal is just the look of old film?

Every date-conditioned experiment in this repo rests on one assumption: that an
image's date is related to *what the scene looked like*. But old photos are also
grayscale, sepia, grainy and low-contrast, and modern ones are colour and sharp.
A model (or a metric, or a probe) can score well on "what year is this?" while
having learned nothing but "brown and grey means old". The literature calls this
the photo-medium shortcut; `docs/literature_review.md` flags it, and
`docs/findings.md` reports that pixel-based monochrome detection already
separates most pre-1945 photos from modern ones.

Nobody has measured how big that shortcut is in *our* data. This module does,
in two ways:

1. **An ablation.** Fit a linear probe that predicts the capture year from cheap
   appearance statistics, then refit it on the same photos after removing colour
   (grayscale) and then after also flattening the tone curve. Whatever disappears
   was colour or tone, and so was shortcut.
   The surviving skill is an **upper bound** on content-driven dating, not a
   measurement of it: grain, softness and scan resolution are medium cues too,
   and no treatment can remove them without destroying the scene as well. On
   synthetic photos that contain *no* content at all, tone matching still leaves
   the probe about a third of its skill (`tests/test_shortcut.py`). Read the last
   row as "at most this much could be content".

2. **A causal test** (:func:`sepia_shift`). Take *modern* photos, apply a
   period-looking sepia/contrast treatment, and ask the already-fitted probe for
   the year again. The scene content is untouched, so any shift in the predicted
   year is caused by the medium alone. Correlational ablations can be argued
   with; this one cannot.

The probe is deliberately weak and linear: the question is how much era
information is *trivially* available, not how well a big model can do.

Leakage matters here. The same photograph often appears several times in
MegaScenes, and photographers shoot bursts from one spot, so a random split puts
near-identical images in both train and test and inflates every number. We reuse
the near-duplicate grouping from :mod:`timemachine.eval.splits`, which was built
for exactly this.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from timemachine.eval.splits import near_duplicate_groups, view_c2ws

# ---------------------------------------------------------------------------
# Appearance statistics
# ---------------------------------------------------------------------------
# Deliberately *only* describes how the photo looks (colour, tone, sharpness,
# grain) and never what is in it. If a probe on these features can date a photo,
# the information is in the medium, not the scene.
FEATURE_NAMES = (
    "saturation_mean", "saturation_p90", "chroma_major", "chroma_minor", "chroma_angle",
    "lum_mean", "lum_std", "lum_p05", "lum_p95", "lum_skew",
    "edge_energy", "high_freq_ratio", "aspect", "log_pixels",
)

# The same features, grouped by which physical cue they describe. Fitting the probe
# on one group at a time decomposes the signal: it answers "what is the date actually
# being read from?" rather than only "is it readable?". Grain and framing are medium
# cues that no image treatment can strip without destroying the scene, so this
# decomposition is the only way to see them separately.
FEATURE_GROUPS = {
    "colour": ("saturation_mean", "saturation_p90", "chroma_major", "chroma_minor", "chroma_angle"),
    "tone": ("lum_mean", "lum_std", "lum_p05", "lum_p95", "lum_skew"),
    "detail": ("edge_energy", "high_freq_ratio"),
    "framing": ("aspect", "log_pixels"),
}

_THUMB = 128  # everything is computed on a thumbnail: fast, and resolution-independent


def _thumbnail(image: Image.Image, size: int = _THUMB) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)


def appearance_features(image: Image.Image) -> dict[str, float]:
    """Appearance-only statistics of one image (see :data:`FEATURE_NAMES`).

    Colour: mean and 90th-percentile saturation, and the spread of the Lab (a, b)
    chroma cloud along its major and minor axes plus its direction. A toned print
    (sepia, cyanotype, plain grey) varies along a single colour axis, so its
    *minor* spread is near zero however strong the tint; the angle says which
    tint it is.

    Tone: brightness mean, spread, 5th/95th percentiles and skew, which capture
    the flat, mid-grey look of an old scan versus a modern photo's deep blacks.

    Detail: an edge-energy measure and the share of energy in fine detail, which
    pick up the softness of old lenses and scans, and film grain.
    """
    w, h = image.size
    thumb = _thumbnail(image)
    rgb = np.asarray(thumb, dtype=np.float32) / 255.0
    hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32) / 255.0
    lab = np.asarray(thumb.convert("LAB"))
    # Pillow stores Lab's a/b channels as signed bytes dressed up as unsigned.
    ab = np.stack([lab[..., 1].view(np.int8), lab[..., 2].view(np.int8)], -1).reshape(-1, 2).astype(np.float64)

    cov = np.cov(ab.T) if len(ab) > 1 else np.zeros((2, 2))
    eigvals, eigvecs = np.linalg.eigh(cov)
    major, minor = np.sqrt(max(eigvals[1], 0.0)), np.sqrt(max(eigvals[0], 0.0))
    # Direction of the dominant colour axis, folded to [0, pi): sepia and cyanotype
    # sit at different angles, and a neutral grey image has no meaningful angle.
    angle = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]) % np.pi) if major > 1e-6 else 0.0

    lum = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum_std = float(lum.std())
    skew = float(((lum - lum.mean()) ** 3).mean() / (lum_std ** 3 + 1e-8))

    # Cheap sharpness/grain proxies: first differences, and how much of that
    # energy sits at the finest scale (compared with a 2x-downsampled copy).
    dx, dy = np.diff(lum, axis=1), np.diff(lum, axis=0)
    edge = float(np.sqrt((dx ** 2).mean() + (dy ** 2).mean()))
    coarse = lum.reshape(lum.shape[0] // 2, 2, lum.shape[1] // 2, 2).mean(axis=(1, 3))
    cdx, cdy = np.diff(coarse, axis=1), np.diff(coarse, axis=0)
    coarse_edge = float(np.sqrt((cdx ** 2).mean() + (cdy ** 2).mean()))

    return {
        "saturation_mean": float(hsv[..., 1].mean()),
        "saturation_p90": float(np.percentile(hsv[..., 1], 90)),
        "chroma_major": major,
        "chroma_minor": minor,
        "chroma_angle": angle,
        "lum_mean": float(lum.mean()),
        "lum_std": lum_std,
        "lum_p05": float(np.percentile(lum, 5)),
        "lum_p95": float(np.percentile(lum, 95)),
        "lum_skew": skew,
        "edge_energy": edge,
        "high_freq_ratio": edge / (coarse_edge + 1e-8),
        "aspect": float(w / h) if h else 1.0,
        "log_pixels": float(np.log10(max(w * h, 1))),
    }


def feature_matrix(rows: Sequence[dict]) -> np.ndarray:
    """Stack per-image feature dicts into an (N, F) matrix in :data:`FEATURE_NAMES` order."""
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=np.float64)


def group_columns(group: str) -> list[int]:
    """Column indices of one :data:`FEATURE_GROUPS` entry in a :func:`feature_matrix`."""
    if group not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group {group!r}; known: {sorted(FEATURE_GROUPS)}")
    return [FEATURE_NAMES.index(name) for name in FEATURE_GROUPS[group]]


# ---------------------------------------------------------------------------
# Image treatments: strip the medium cues, one at a time
# ---------------------------------------------------------------------------
TREATMENTS = ("original", "grayscale", "tone_matched")


def to_grayscale(image: Image.Image) -> Image.Image:
    """Drop all colour. Removes the sepia/colour cue, keeps tone and detail.

    Worth knowing: this on its own is a **weak** defence against the medium
    shortcut. An old print is flat and mid-grey as well as brown, and that tone
    survives desaturation, so a probe can score just as well on grayscaled
    photos (`tests/test_shortcut.py`). Grayscale evaluation, which the
    literature review suggests as a control, is therefore not sufficient by
    itself; the tone has to be matched too.
    """
    return image.convert("L").convert("RGB")


def to_tone_matched(image: Image.Image) -> Image.Image:
    """Grayscale, then force a common tone curve by histogram equalization.

    This is the strictest condition we can apply without altering the scene: no
    colour, and no information in how bright or contrasty the photo is either.
    It does *not* remove grain, softness or scan resolution, which are medium
    cues as well, so skill measured here is an upper bound on content-driven
    dating rather than a measurement of it (see the module docstring).
    """
    return ImageOps.equalize(image.convert("L")).convert("RGB")


def apply_sepia(image: Image.Image, strength: float = 1.0, contrast: float = 0.75) -> Image.Image:
    """Make a modern photo *look* like an old print, without touching its content.

    Grayscale, flatten the contrast towards mid-grey (old prints rarely hold deep
    blacks or clean whites), then tint towards warm brown. ``strength`` 0 leaves
    the image grayscale and 1 applies the full tint. This is only a plausible
    imitation of an albumen print, not a calibrated film model, which is enough
    for a causal test: we change the look and nothing else.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    grey = 0.5 + (grey - 0.5) * contrast
    tint = np.array([1.07, 0.74, 0.43], dtype=np.float32)       # warm brown
    toned = grey[..., None] * (1.0 - strength + strength * tint)
    return Image.fromarray((np.clip(toned, 0, 1) * 255).astype(np.uint8), mode="RGB")


TREATMENT_FUNCS = {"original": lambda im: im, "grayscale": to_grayscale, "tone_matched": to_tone_matched}


# ---------------------------------------------------------------------------
# A deliberately simple linear probe
# ---------------------------------------------------------------------------
class YearProbe:
    """Ridge regression from appearance features to capture year.

    Closed form, so there is nothing to tune and no randomness: the same data
    always gives the same answer, which matters when the whole point is to
    compare conditions against each other. Features are standardized on the
    training set, and the intercept is not penalized.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> "YearProbe":
        """Fit, optionally with per-image weights (see :func:`balanced_weights`).

        Weighting matters more than it looks. Our scenes are ~75% modern, so an
        unweighted fit is dominated by one tight clump of recent photos and
        effectively ignores the old tail we actually care about.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(X) < 2:
            raise ValueError("Need at least two training images.")
        w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=np.float64)
        # Normalize to sum to N, not 1, so that `alpha` means the same thing whether
        # or not weights are given (the Gram matrix keeps its unweighted magnitude).
        w = w / w.sum() * len(w)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0        # a constant feature carries nothing
        Z = (X - self.mean_) / self.scale_
        self.intercept_ = float((w @ y) / w.sum())
        # (Z'WZ + alpha I)^-1 Z'W(y - mean). solve() beats inv() for conditioning.
        gram = Z.T @ (Z * w[:, None]) + self.alpha * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(gram, Z.T @ (w * (y - self.intercept_)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Probe is not fitted.")
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_


def balanced_weights(years: np.ndarray, old_before: float = 1945.0) -> np.ndarray:
    """Weights that give the old and modern photos equal say.

    Our view tables are roughly three-quarters modern, so an unweighted fit is
    really a fit to recent photos with the old ones as rounding error.
    """
    years = np.asarray(years, dtype=np.float64)
    old = years < old_before
    w = np.ones(len(years))
    if old.any() and (~old).any():
        w[old] = 0.5 / old.sum()
        w[~old] = 0.5 / (~old).sum()
    return w / w.sum() * len(w)


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, from ranks (handles ties, no sklearn needed).

    Reads as: the chance that a randomly chosen old photo scores higher than a
    randomly chosen modern one. 0.5 is coin-flipping, 1.0 is perfect. Unlike
    accuracy it does not flatter a probe just because one class is rare, which
    is why it is the headline number here.
    """
    scores, positive = np.asarray(scores, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average the ranks of tied scores, or ties would count as wins
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(scores) + 1):
        if i == len(scores) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_interval(scores: np.ndarray, positive: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for :func:`roc_auc`.

    Our old photos are few (a single scene can leave fewer than 20 in a test
    split), so a bare AUC reads as more precise than it is. Resampling is done
    within each class, which keeps the class sizes fixed and matches how the
    split was drawn.
    """
    scores, positive = np.asarray(scores, dtype=np.float64), np.asarray(positive, dtype=bool)
    pos, neg = np.flatnonzero(positive), np.flatnonzero(~positive)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws[i] = roc_auc(scores[idx], positive[idx])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score_era(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray,
              alpha: float = 1.0, old_before: float = 1945.0) -> dict:
    """Can the probe tell an old photo from a modern one, from its look alone?

    A least-squares classifier on a balanced +1 (old) / -1 (modern) target. This,
    not the year regression, is the question that matters: our year distribution
    is bimodal and three-quarters modern, so regressing the year punishes a probe
    that has in fact learned the old/modern split perfectly well.

    Reported as AUC and balanced accuracy (the mean of the two per-class
    recalls), neither of which can be gamed by always answering "modern".
    """
    train_old, test_old = np.asarray(train_y) < old_before, np.asarray(test_y) < old_before
    if not train_old.any() or train_old.all():
        return {"auc": float("nan"), "auc_lo": float("nan"), "auc_hi": float("nan"),
                "balanced_accuracy": float("nan"),
                "n_old": int(test_old.sum()), "n_modern": int((~test_old).sum()), "probe": None}
    probe = YearProbe(alpha).fit(train_X, np.where(train_old, 1.0, -1.0),
                                 balanced_weights(train_y, old_before))
    scores = probe.predict(test_X)
    recall_old = float((scores[test_old] > 0).mean()) if test_old.any() else float("nan")
    recall_modern = float((scores[~test_old] <= 0).mean()) if (~test_old).any() else float("nan")
    auc_lo, auc_hi = auc_interval(scores, test_old)
    return {
        "auc": roc_auc(scores, test_old), "auc_lo": auc_lo, "auc_hi": auc_hi,
        "balanced_accuracy": float(np.nanmean([recall_old, recall_modern])),
        "recall_old": recall_old, "recall_modern": recall_modern,
        "n_old": int(test_old.sum()), "n_modern": int((~test_old).sum()),
        "probe": probe,
    }


def baseline_mae(train_years: np.ndarray, test_years: np.ndarray) -> float:
    """MAE of the best constant guess (the training median). The probe must beat this."""
    return float(np.abs(np.asarray(test_years) - float(np.median(train_years))).mean())


def score_probe(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray, test_y: np.ndarray,
                alpha: float = 1.0, old_before: float = 1945.0) -> dict:
    """Fit on train, score on test. MAE in years, R^2, and old-vs-modern accuracy.

    ``skill`` is how much of the constant-guess error the probe removes:
    0 means it learned nothing, 1 means perfect. It is the honest headline here,
    because raw MAE depends on how spread out the years happen to be.
    """
    # The year fit stays unweighted and is reported only as a secondary number;
    # balancing belongs in the old/modern classifier below, where the class sizes
    # are what distort the result.
    probe = YearProbe(alpha).fit(train_X, train_y)
    pred = probe.predict(test_X)
    test_y = np.asarray(test_y, dtype=np.float64)
    mae = float(np.abs(pred - test_y).mean())
    base = baseline_mae(train_y, test_y)
    era = score_era(train_X, train_y, test_X, test_y, alpha, old_before)
    return {
        "n_train": int(len(train_X)), "n_test": int(len(test_X)),
        "auc": era["auc"], "auc_lo": era["auc_lo"], "auc_hi": era["auc_hi"],
        "balanced_accuracy": era["balanced_accuracy"],
        "recall_old": era.get("recall_old", float("nan")),
        "recall_modern": era.get("recall_modern", float("nan")),
        "n_old": era["n_old"], "n_modern": era["n_modern"],
        "mae": mae, "baseline_mae": base,
        "skill": float(1.0 - mae / base) if base > 0 else float("nan"),
        "old_share": float((test_y < old_before).mean()),
        "probe": probe, "era_probe": era["probe"],
    }


def sepia_shift(year_probe: YearProbe, era_probe: YearProbe | None, modern_features: np.ndarray,
                sepia_features: np.ndarray, years: np.ndarray) -> dict:
    """The causal test: tone a modern photo and ask the probes to date it again.

    ``modern_features`` and ``sepia_features`` are the same photographs before and
    after :func:`apply_sepia`, row for row. The scene is identical, so the whole
    change is caused by the medium.

    The headline is ``flip_rate``: the share of modern photos that the *old/modern
    classifier* moves across its decision boundary. That classifier is the one we
    trust (see :func:`score_era`), and a rate well above zero says the model of
    "old" is really a model of film stock.

    The year-regression numbers are kept as secondary colour. Read them loosely:
    across several scenes that probe is pulled towards the middle of a bimodal
    year distribution, so its absolute shifts understate the effect.
    """
    before, after = year_probe.predict(modern_features), year_probe.predict(sepia_features)
    shift = after - before
    out = {
        "n": int(len(shift)),
        "median_year_before": float(np.median(before)),
        "median_year_after": float(np.median(after)),
        "median_shift": float(np.median(shift)),
        "mean_shift": float(shift.mean()),
        "share_moved_before_1945": float(((before >= 1945) & (after < 1945)).mean()),
        "true_median_year": float(np.median(years)),
        "flip_rate": float("nan"), "era_score_shift": float("nan"), "called_old_before": float("nan"),
    }
    if era_probe is not None:
        era_before, era_after = era_probe.predict(modern_features), era_probe.predict(sepia_features)
        # The classifier scores +1 for old and -1 for modern, so 0 is the boundary.
        out["called_old_before"] = float((era_before > 0).mean())
        out["flip_rate"] = float(((era_before <= 0) & (era_after > 0)).mean())
        out["era_score_shift"] = float(np.median(era_after - era_before))
    return out


# ---------------------------------------------------------------------------
# Loading view tables
# ---------------------------------------------------------------------------
VIEW_FILES = ("views_clean.parquet", "views.parquet")   # prefer the audited table

def load_views(folders: list[Path], max_views: int, seed: int) -> pd.DataFrame:
    """Concatenate the view tables of several single-scene dataset folders."""
    frames = []
    for folder in folders:
        path = next((folder / name for name in VIEW_FILES if (folder / name).exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"{folder} has none of {VIEW_FILES}. Build it with scripts/build_scene_dataset.py.")
        views = pd.read_parquet(path)
        if "image_path" not in views:
            raise ValueError(f"{path} has no image_path column; the images were never cached.")
        views = views.assign(dataset=folder.name,
                             image_path=resolve_image_paths(views["image_path"], folder))
        frames.append(views)
    views = pd.concat(frames, ignore_index=True)
    views = views[views["date_mid"].notna() & views["image_path"].notna()].reset_index(drop=True)
    if max_views and len(views) > max_views:
        # Subsample proportionally within each scene, so one big scene cannot dominate
        # the probe. Done with explicit row indices rather than groupby().apply(),
        # which silently drops the grouping column on some pandas versions.
        rng = np.random.default_rng(seed)
        share = max_views / len(views)
        keep: list[int] = []
        for name in views["dataset"].unique():
            rows = np.flatnonzero((views["dataset"] == name).to_numpy())
            take = min(len(rows), max(2, int(round(len(rows) * share))))
            keep.extend(rng.choice(rows, size=take, replace=False).tolist())
        views = views.iloc[sorted(keep)].reset_index(drop=True)
    return views


def resolve_image_paths(paths: pd.Series, folder: Path) -> pd.Series:
    """Make ``image_path`` usable even if the dataset folder was copied here from elsewhere.

    View tables store absolute paths from the machine that built them (usually
    the cluster), so a folder copied to a laptop points at directories that do
    not exist. Any path that is missing is retried relative to the dataset
    folder, to its ``images/`` subfolder, and to the current directory, matching
    on the file name.
    """
    resolved, missing = [], 0
    for raw in paths.astype(str):
        candidate = Path(raw)
        if not candidate.exists():
            name = candidate.name
            for base in (folder, folder / "images", folder.parent, Path.cwd()):
                if (base / name).exists():
                    candidate = base / name
                    break
            else:
                missing += 1
        resolved.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            f"{missing} of {len(resolved)} images under {folder} could not be found.\n"
            "The view table stores absolute paths from the machine that built it. Either run this "
            "where those paths exist, or copy the cached images next to the table (in "
            f"{folder}/images/) so they can be found by name.")
    return pd.Series(resolved, index=paths.index)


def grouped_split(views: pd.DataFrame, test_fraction: float, seed: int) -> np.ndarray:
    """Boolean train mask that keeps near-duplicate views on the same side.

    Each scene is grouped on its own (poses from different scenes are in
    different coordinate frames and must never be compared).
    """
    groups = np.zeros(len(views), dtype=np.int64)
    offset = 0
    for name in views["dataset"].unique():
        idx = np.flatnonzero((views["dataset"] == name).to_numpy())
        part = views.iloc[idx]
        try:
            local = near_duplicate_groups(part["date_mid"].to_numpy(float), view_c2ws(part))
        except Exception:                      # a table without full pose columns still works
            local = np.arange(len(part))
        groups[idx] = local + offset
        offset += int(local.max()) + 1

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    test_groups = set(rng.permutation(unique)[: max(1, int(round(len(unique) * test_fraction)))])
    return ~np.isin(groups, list(test_groups)), groups


def features_for(paths: list[str], treatment: str) -> np.ndarray:
    """Appearance features of every image under one treatment."""
    func = TREATMENT_FUNCS[treatment]
    rows = []
    for path in paths:
        with Image.open(path) as im:
            rows.append(appearance_features(func(im.convert("RGB"))))
    return feature_matrix(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def group_skill_table(train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray,
                      test_y: np.ndarray, alpha: float = 1.0) -> pd.DataFrame:
    """Skill of a probe fitted on each feature group alone, plus all groups together.

    A high score for ``colour`` means the date is being read off the film stock.
    A high score for ``framing`` means it is being read off the print format
    (old prints are often odd-shaped, and scans of them are small). Neither is
    scene content. The groups overlap in what they explain, so their scores do
    not add up to the combined one.
    """
    rows = []
    for group in (*FEATURE_GROUPS, "all"):
        cols = list(range(train_X.shape[1])) if group == "all" else group_columns(group)
        s = score_probe(train_X[:, cols], train_y, test_X[:, cols], test_y, alpha)
        rows.append({"features": group, "n_features": len(cols), "auc": s["auc"],
                     "balanced_accuracy": s["balanced_accuracy"], "mae": s["mae"], "skill": s["skill"]})
    return pd.DataFrame(rows)


def results_table(scores: dict[str, dict]) -> pd.DataFrame:
    """One row per treatment, in :data:`TREATMENTS` order."""
    rows = [{"treatment": name, **{k: v for k, v in s.items() if not k.endswith("probe")}}
            for name, s in scores.items()]
    table = pd.DataFrame(rows)
    order = [t for t in TREATMENTS if t in set(table["treatment"])]
    table["treatment"] = pd.Categorical(table["treatment"], categories=order, ordered=True)
    return table.sort_values("treatment").reset_index(drop=True)


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def render_report(table: pd.DataFrame, shift: dict | None, n_scenes: int,
                  groups: pd.DataFrame | None = None, tol_note: str = "") -> str:
    """Markdown summary, ready to paste into docs/findings.md."""
    first = table.iloc[0] if len(table) else None
    lines = [
        "## How much of our date signal is just the photo medium?",
        f"*Linear probe on appearance-only statistics (colour, tone, grain, framing), {n_scenes} scene(s), "
        f"near-duplicate-safe split. {tol_note}*",
        "",
        "A probe that only sees *how the photo looks* should not be able to date it. Each row removes "
        "one more medium cue; whatever survives is not colour or tone.",
        "",
        "**AUC** is the chance that a randomly chosen old photo scores as older than a randomly chosen "
        "modern one: 0.5 is coin-flipping, 1.0 is perfect. It is the headline because it cannot be "
        "gamed by always answering \"modern\", which plain accuracy can when most photos are recent.",
        "",
        "| treatment | n test | old / modern | AUC | 95% CI | balanced accuracy | recall old | recall modern |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.treatment} | {r.n_test} | {r.n_old} / {r.n_modern} | {r.auc:.3f} | "
                     f"{r.auc_lo:.2f}-{r.auc_hi:.2f} | {_pct(r.balanced_accuracy)} | "
                     f"{_pct(r.recall_old)} | {_pct(r.recall_modern)} |")
    if first is not None and first.n_old < 40:
        lines += ["", f"**Only {first.n_old} old photos are in the test split**, so the intervals are wide. "
                      "Treat a single scene as indicative and re-run across several scenes before "
                      "quoting a number."]
    lines += [
        "", "`tone_matched` removes colour and tone but **not** grain, softness or scan resolution, "
            "which are medium cues too. Its AUC is therefore an upper bound on how much of the date is "
            "readable from scene content, not a measurement of it.",
    ]
    if groups is not None and len(groups):
        lines += ["", "### Which cue carries the date?",
                  "*Probe fitted on one group of features at a time, on the untreated photos.*", "",
                  "| features | n | AUC | balanced accuracy |", "|---|---|---|---|"]
        lines += [f"| {r.features} | {r.n_features} | {r.auc:.3f} | {_pct(r.balanced_accuracy)} |"
                  for r in groups.itertuples()]
        lines += ["", "`colour`, `tone`, `detail` and `framing` are all properties of the *photograph*, "
                      "not of the scene. Skill here is shortcut, whichever group it comes from."]
    if shift:
        lines += [
            "", "### Causal check: sepia-toning modern photos",
            f"*{shift['n']} modern photos (true median {shift['true_median_year']:.0f}), "
            "grayscaled, flattened and warm-toned. The scene content is untouched, so any shift is "
            "caused by the medium alone.*", "",
            f"- The old/modern classifier calls {_pct(shift['called_old_before'])} of them old "
            f"untreated, and **{_pct(shift['flip_rate'])} more flip to 'old' once toned** "
            f"(median score shift {shift['era_score_shift']:+.2f}).",
            f"- The year probe moves them from {shift['median_year_before']:.0f} to "
            f"{shift['median_year_after']:.0f} (median **{shift['median_shift']:+.1f} years**), "
            "though that probe is the less reliable of the two across scenes.",
        ]
    if first is not None:
        lines += [
            "", "### A note on the year-regression columns",
            "`treatments.csv` also carries `mae` and `skill` from a probe regressing the *year*. "
            "Read those with care, and prefer the AUC above: our year distribution is bimodal and "
            "about three-quarters modern, so a least-squares fit is pulled towards the mean while the "
            "constant-guess baseline sits at the median. On Notre-Dame those are ~20 years apart, "
            "which makes `skill` go negative even for a probe that separates old from modern well. "
            "The split is the real question; the exact year is not.",
        ]
    return "\n".join(lines) + "\n"
