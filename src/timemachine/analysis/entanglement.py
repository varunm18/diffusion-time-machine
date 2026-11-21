"""Is the "how it was lit" direction separable from the "when it was taken" direction?

Why this matters. `timemachine.model.appearance_film` conditions the model on a
per-view appearance vector so that lighting, weather and photographic medium have
somewhere to live other than the date. But those statistics were chosen *because*
they predict era (AUC 0.89, `docs/findings.md`), which creates a hazard the
original design glossed over: handing the model a near-perfect era signal through
the appearance channel may teach it to ignore the date even more than before.

The fix, if it is needed, is to condition on the part of appearance that carries
no era information. This module measures whether that part exists and is big
enough to be worth anything:

1. :func:`era_direction` finds the single direction in appearance space that best
   separates old from modern photos.
2. :func:`residualize` projects it out, leaving features that are orthogonal to it.
3. :func:`residual_report` checks what that cost and what survived - whether era
   really is gone (refitting a fresh classifier on the residual, not assuming),
   how much variance is left, and which statistics were most era-aligned.
4. With labels for something else we want the model to see - day versus night,
   say - :func:`label_separability` asks whether the residual still carries *it*.

Read the result as a two-by-two. Era gone and lighting kept means the
factorization is clean and the model should be conditioned on the residual. Both
gone means the two are entangled in these features, and absorbing lighting
necessarily leaks era, so the appearance pathway cannot be made era-free this way
and Phase C should be read with that in mind.

Nothing here is circular: the lighting labels come from a human (or from any
source outside these statistics), never from the statistics themselves. Defining
"night" as "low ``lum_mean``" and then reporting that ``lum_mean`` predicts night
would measure nothing at all.
"""
from __future__ import annotations

import base64
import html
import io
from collections.abc import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from timemachine.analysis.shortcut import (
    FEATURE_GROUPS, FEATURE_NAMES, YearProbe, auc_interval, balanced_weights, roc_auc,
)


def standardize(X: np.ndarray, mean: np.ndarray | None = None,
                scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-variance columns, so one large-valued statistic cannot dominate a direction."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0) if mean is None else mean
    scale = X.std(axis=0) if scale is None else scale
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (X - mean) / scale, mean, scale


def era_direction(X: np.ndarray, years: np.ndarray, old_before: float = 1945.0,
                  alpha: float = 1.0) -> np.ndarray:
    """Unit vector in *standardized* appearance space pointing from modern towards old.

    This is the balanced least-squares classifier's weight vector - the same probe
    used to measure the shortcut, so the direction removed below is exactly the one
    that made the date readable.
    """
    Z, _, _ = standardize(X)
    old = np.asarray(years) < old_before
    if not old.any() or old.all():
        raise ValueError("Need both old and modern photos to define an era direction.")
    return _fit_direction(Z, old, years, old_before, alpha)


def _fit_direction(Z: np.ndarray, old: np.ndarray, years: np.ndarray, old_before: float,
                   alpha: float) -> np.ndarray:
    """Unit era direction for already-standardized features."""
    probe = YearProbe(alpha).fit(Z, np.where(old, 1.0, -1.0), balanced_weights(years, old_before))
    w = probe.coef_ / probe.scale_          # undo the probe's own internal scaling
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("Era direction is degenerate (the probe learned nothing).")
    return w / norm


def residualize(X: np.ndarray, direction: np.ndarray, mean: np.ndarray | None = None,
                scale: np.ndarray | None = None) -> np.ndarray:
    """Appearance features with ``direction`` projected out (returned standardized).

    ``z - (z . d) d``. Statistics that lie along the era direction lose their
    content; statistics orthogonal to it are untouched.
    """
    Z, _, _ = standardize(X, mean, scale)
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    return Z - np.outer(Z @ d, d)


def _era_auc(X: np.ndarray, years: np.ndarray, train: np.ndarray, old_before: float,
             alpha: float) -> tuple[float, float, float]:
    """AUC (+ interval) of a *freshly fitted* era classifier on held-out rows."""
    old = np.asarray(years) < old_before
    if not old[train].any() or old[train].all() or not old[~train].any() or old[~train].all():
        return float("nan"), float("nan"), float("nan")
    probe = YearProbe(alpha).fit(X[train], np.where(old[train], 1.0, -1.0),
                                 balanced_weights(years[train], old_before))
    scores = probe.predict(X[~train])
    lo, hi = auc_interval(scores, old[~train])
    return roc_auc(scores, old[~train]), lo, hi


def era_subspace(X: np.ndarray, years: np.ndarray, train: np.ndarray, old_before: float = 1945.0,
                 alpha: float = 1.0, target_auc: float = 0.60, max_dims: int = 8) -> dict:
    """Remove era directions one at a time until a refitted classifier can no longer find era.

    One direction is rarely enough. A sepia cast moves saturation, both chroma
    spreads and the tone statistics together, so era lives in a *subspace*, not on
    an axis - project out the best direction and a refit still finds it in what is
    left. The useful question is therefore not "does removing it work?" but **how
    many of the 14 dimensions must be destroyed before the date is gone, and how
    much of appearance is left afterwards?** A small number means the pathway can
    be made era-free cheaply; a large one means appearance and era are close to the
    same thing in these statistics.

    Stops when a freshly fitted classifier on the residual falls below
    ``target_auc``, or after ``max_dims`` removals.
    """
    Z, mean, scale = standardize(X)
    old = np.asarray(years) < old_before
    directions: list[np.ndarray] = []
    trace = []
    residual = Z.copy()

    auc, lo, hi = _era_auc(residual, years, train, old_before, alpha)
    trace.append({"dims_removed": 0, "era_auc": auc, "ci_low": lo, "ci_high": hi,
                  "variance_kept": 1.0})
    while auc >= target_auc and len(directions) < max_dims:
        d = _fit_direction(residual[train], old[train], years[train], old_before, alpha)
        for prev in directions:                     # keep the basis orthonormal
            d = d - (d @ prev) * prev
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            break                                   # nothing new left to remove
        d = d / norm
        directions.append(d)
        residual = residual - np.outer(residual @ d, d)
        auc, lo, hi = _era_auc(residual, years, train, old_before, alpha)
        trace.append({"dims_removed": len(directions), "era_auc": auc, "ci_low": lo, "ci_high": hi,
                      "variance_kept": float(residual.var(axis=0).sum() / max(Z.var(axis=0).sum(), 1e-12))})
    return {"directions": np.array(directions) if directions else np.zeros((0, X.shape[1])),
            "trace": pd.DataFrame(trace), "mean": mean, "scale": scale,
            "era_removed": bool(auc < target_auc), "target_auc": target_auc}


def project_out(X: np.ndarray, directions: np.ndarray, mean: np.ndarray | None = None,
                scale: np.ndarray | None = None) -> np.ndarray:
    """Standardized features with every row of ``directions`` projected out."""
    Z, _, _ = standardize(X, mean, scale)
    for d in np.atleast_2d(directions):
        if np.linalg.norm(d) > 1e-12:
            d = d / np.linalg.norm(d)
            Z = Z - np.outer(Z @ d, d)
    return Z


def residual_report(X: np.ndarray, years: np.ndarray, train: np.ndarray,
                    old_before: float = 1945.0, alpha: float = 1.0,
                    target_auc: float = 0.60, max_dims: int = 8) -> dict:
    """How much of appearance must be destroyed to remove the date, and what survives."""
    sub = era_subspace(X, years, train, old_before, alpha, target_auc, max_dims)
    trace = sub["trace"]
    first, last = trace.iloc[0], trace.iloc[-1]
    primary = sub["directions"][0] if len(sub["directions"]) else np.zeros(X.shape[1])
    loading = pd.DataFrame({"feature": list(FEATURE_NAMES), "era_loading": np.abs(primary)})
    return {
        "directions": sub["directions"], "direction": primary,
        "mean": sub["mean"], "scale": sub["scale"],
        "trace": trace,
        "dims_removed": int(last["dims_removed"]), "n_features": X.shape[1],
        "era_removed": sub["era_removed"], "target_auc": target_auc,
        "era_auc_full": float(first["era_auc"]), "era_auc_full_ci": (first["ci_low"], first["ci_high"]),
        "era_auc_residual": float(last["era_auc"]),
        "era_auc_residual_ci": (last["ci_low"], last["ci_high"]),
        "variance_kept": float(last["variance_kept"]),
        "loadings": loading.sort_values("era_loading", ascending=False).reset_index(drop=True),
        "n_train": int(train.sum()), "n_test": int((~train).sum()),
        "n_old_test": int((np.asarray(years)[~train] < old_before).sum()),
    }


def label_separability(X: np.ndarray, labels: np.ndarray, train: np.ndarray,
                       directions: np.ndarray, alpha: float = 1.0) -> dict:
    """Does some other property (lighting, say) survive removing the era direction?

    ``labels`` is a boolean array from outside these statistics - hand labels, or
    any independent source. Using a label derived from the features themselves
    would make this meaningless.
    """
    labels = np.asarray(labels, dtype=bool)
    Z, mean, scale = standardize(X)
    residual = project_out(X, np.atleast_2d(directions), mean, scale)

    def fit_auc(features):
        if not labels[train].any() or labels[train].all() or not labels[~train].any() or labels[~train].all():
            return float("nan")
        w = np.where(labels[train], 0.5 / labels[train].sum(), 0.5 / (~labels[train]).sum())
        probe = YearProbe(alpha).fit(features[train], np.where(labels[train], 1.0, -1.0), w)
        return roc_auc(probe.predict(features[~train]), labels[~train])

    full, residual_auc = fit_auc(Z), fit_auc(residual)
    return {
        "auc_full": full, "auc_residual": residual_auc,
        "retained": float(residual_auc - 0.5) / max(full - 0.5, 1e-9) if full > 0.5 else float("nan"),
        "n_positive": int(labels[~train].sum()), "n_negative": int((~labels[~train]).sum()),
    }


def group_alignment(direction: np.ndarray) -> pd.DataFrame:
    """How much of the era direction sits in each feature group (shares, summing to 1)."""
    d2 = np.asarray(direction, dtype=np.float64) ** 2
    rows = [{"features": g, "share_of_era_direction": float(
        d2[[FEATURE_NAMES.index(n) for n in names]].sum() / max(d2.sum(), 1e-12))}
        for g, names in FEATURE_GROUPS.items()]
    return pd.DataFrame(rows).sort_values("share_of_era_direction", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Collecting lighting labels
# ---------------------------------------------------------------------------
LIGHTING_CLASSES = ("day", "night", "unclear")

# Era buckets for spreading a labelling sheet. Defined here rather than imported so
# this analysis does not move when the shortcut probe's reporting changes.
ERA_EDGES = (1900.0, 1945.0, 1990.0)
ERA_LABELS = ("pre-1900", "1900-1944", "1945-1989", "1990+")


def era_bucket(years) -> np.ndarray:
    """Era label (see :data:`ERA_LABELS`) for an array-like of years."""
    idx = np.digitize(np.atleast_1d(np.asarray(years, dtype=float)), ERA_EDGES)
    return np.asarray(ERA_LABELS, dtype=object)[idx]

_SHEET_CSS = """
body{font:15px/1.4 system-ui,sans-serif;margin:0;padding:16px;max-width:1000px;margin-inline:auto;
color:#1c1c1c;background:#fafafa}
h1{font-size:20px;margin:0 0 8px}.help{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;
margin-bottom:16px}.help li{margin:3px 0}code{background:#eee;padding:1px 4px;border-radius:3px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}
.cell{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}
.cell img{width:100%;height:130px;object-fit:cover;border-radius:4px;background:#eee}
.btns{display:flex;gap:4px;margin-top:6px}
.btns button{flex:1;font:13px system-ui;padding:5px 0;border:1px solid #bbb;background:#f4f4f4;
border-radius:5px;cursor:pointer;color:#1c1c1c}
.btns button[aria-pressed="true"]{background:#222;color:#fff;border-color:#222}
.bar{position:sticky;top:0;background:#fafafa;padding:8px 0;display:flex;gap:12px;align-items:center;z-index:1}
#dl{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid #444;background:#222;color:#fff;cursor:pointer}
@media(prefers-color-scheme:dark){body{background:#161616;color:#eee}.help,.cell{background:#212121;border-color:#383838}
code{background:#333}.bar{background:#161616}.btns button{background:#2c2c2c;color:#eee;border-color:#555}
.btns button[aria-pressed="true"]{background:#eee;color:#111}}
"""

_SHEET_JS = """
const KEY="lighting:"+document.title;
const store={get(){try{return JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){return{}}},
             set(v){try{localStorage.setItem(KEY,JSON.stringify(v))}catch(e){}}};
const saved=store.get(), cells=document.querySelectorAll(".cell");
function update(){let n=0;cells.forEach(c=>{if(c.dataset.choice)n++});
  document.getElementById("count").textContent=n+" / "+cells.length+" labelled";}
cells.forEach(c=>{
  if(saved[c.dataset.id]){c.dataset.choice=saved[c.dataset.id];}
  c.querySelectorAll("button").forEach(b=>{
    if(c.dataset.choice===b.dataset.v)b.setAttribute("aria-pressed","true");
    b.addEventListener("click",()=>{
      c.dataset.choice=b.dataset.v;
      c.querySelectorAll("button").forEach(o=>o.setAttribute("aria-pressed",String(o===b)));
      const s=store.get();s[c.dataset.id]=b.dataset.v;store.set(s);update();});});});
update();
const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
document.getElementById("dl").addEventListener("click",()=>{
  const lines=["id,lighting"];
  cells.forEach(c=>lines.push([c.dataset.id,c.dataset.choice||""].map(q).join(",")));
  const blob=new Blob(["\ufeff"+lines.join("\r\n")],{type:"text/csv;charset=utf-8"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="lighting_labels.csv";a.click();});
"""


def render_lighting_sheet(ids: Sequence[str], thumbnails: Sequence[str],
                          page_title: str = "Lighting labels") -> str:
    """A self-contained HTML page for labelling day vs night.

    Thumbnails are embedded as data URIs, so the page works with no network and no
    local server - it can be copied off the cluster and opened anywhere. The sheet
    shows no dates and no appearance statistics: a labeller who could see either
    would produce labels contaminated by the very thing we are testing against.
    """
    if len(ids) != len(thumbnails):
        raise ValueError("ids and thumbnails must be the same length")
    cells = []
    for item_id, uri in zip(ids, thumbnails):
        buttons = "".join(f'<button type="button" data-v="{c}" aria-pressed="false">{c}</button>'
                          for c in LIGHTING_CLASSES)
        cells.append(f'<div class="cell" data-id="{html.escape(str(item_id), quote=True)}">'
                     f'<img loading="lazy" alt="" src="{uri}">'
                     f'<div class="btns">{buttons}</div></div>')
    help_html = ("<div class='help'><b>How to label.</b> For each photo, say whether it was taken in "
                 "<code>day</code> or <code>night</code> light; <code>unclear</code> if you cannot tell "
                 "(heavy fog, interiors, extreme close-ups).<ul>"
                 "<li>Judge the <i>light</i>, not the age of the photo. A dim, flat old print taken at "
                 "noon is <code>day</code>.</li>"
                 "<li>Dusk and dawn: pick whichever is closer, or <code>unclear</code>.</li>"
                 "<li>The sheet deliberately shows no dates. Do not look them up.</li></ul></div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(page_title)}</title><style>{_SHEET_CSS}</style></head><body>"
            f"<h1>{html.escape(page_title)}</h1>{help_html}"
            f"<div class='bar'><button id='dl' type='button'>Download lighting_labels.csv</button>"
            f"<span id='count'></span></div>"
            f"<div class='grid'>{''.join(cells)}</div><script>{_SHEET_JS}</script></body></html>")


def era_stratified_sample(views: pd.DataFrame, n: int, seed: int = 0,
                          year_column: str = "date_mid") -> pd.DataFrame:
    """Draw ``n`` views spread evenly over eras.

    This is not a nicety. Our view tables are ~75% modern, so a plain random sheet
    would be almost all recent photos - and if nearly every night photo on it is
    also modern, "night" and "modern" become the same label and the separability
    result means nothing. Spreading over eras breaks that confound in the draw
    itself, before any labelling happens.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    eras = era_bucket(views[year_column].to_numpy(float))
    buckets = {e: np.flatnonzero(eras == e) for e in np.unique(eras)}
    chosen: list[int] = []
    limit = min(n, len(views))
    while len(chosen) < limit:
        progressed = False
        for key in rng.permutation(list(buckets)):
            pool = buckets[key]
            if len(chosen) >= limit or not len(pool):
                continue
            pick = int(rng.choice(pool))
            buckets[key] = pool[pool != pick]
            chosen.append(pick)
            progressed = True
        if not progressed:
            break
    return views.iloc[sorted(chosen)].reset_index(drop=True)


def thumbnail_uri(path: str, size: int = 180, quality: int = 70) -> str:
    """A small JPEG of one image as a ``data:`` URI, so the sheet needs no files beside it."""
    with Image.open(path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((size, size), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        thumb.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def parse_lighting_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(is_night, labelled)`` from a frame with ``id`` and ``lighting`` columns."""
    values = frame["lighting"].astype(str).str.strip().str.lower()
    labelled = values.isin(("day", "night")).to_numpy()
    return (values == "night").to_numpy(), labelled


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def verdict(report: dict, lighting: dict | None) -> str:
    """The conclusion, stated for whichever of the three cases actually happened.

    The earlier version collapsed this to two and could report "lighting does not
    survive" when lighting had survived perfectly and it was *era* that remained.
    """
    if not report["era_removed"]:
        return ("**Era could not be removed.** Even after destroying "
                f"{report['dims_removed']} of {report['n_features']} dimensions a freshly fitted "
                f"classifier still reads era at AUC {report['era_auc_residual']:.2f}. Era is spread "
                "across this whole space rather than sitting in a few directions, so there is no "
                "era-free appearance vector to condition on. Phase C should be read knowing that "
                "the appearance pathway necessarily carries the date.")
    kept = f"{report['dims_removed']} of {report['n_features']} dimensions removed, "\
           f"{_pct(report['variance_kept'])} of the variance kept"
    if lighting is None:
        return (f"**Era can be removed** ({kept}). Whether anything worth conditioning on survives "
                "is unanswered until lighting labels are supplied.")
    if lighting["auc_residual"] > 0.65:
        return (f"**Separable** ({kept}). Lighting survives the removal "
                f"(AUC {lighting['auc_residual']:.2f}) while era does not, so the model can be "
                "conditioned on the residual and get the look without the date.")
    if lighting["auc_full"] <= 0.65:
        return ("**Inconclusive.** Lighting was not readable from these statistics even before the "
                f"removal (AUC {lighting['auc_full']:.2f}), so its loss afterwards says nothing. "
                "Either the labels are noisy or these features do not describe lighting.")
    return (f"**Entangled** ({kept}). Lighting was readable before the removal "
            f"(AUC {lighting['auc_full']:.2f}) and is not after ({lighting['auc_residual']:.2f}), so "
            "in these statistics absorbing the look necessarily leaks the era. The appearance "
            "pathway cannot be made era-free this way.")


def render_report(report: dict, lighting: dict | None = None, n_scenes: int = 0) -> str:
    """Markdown summary, ready to paste into docs/findings.md."""
    lines = [
        "## Can appearance be conditioned on without handing over the date?",
        f"*{report['n_train']} train / {report['n_test']} test views "
        f"({report['n_old_test']} old in test), {n_scenes} scene(s).*",
        "",
        "The appearance vector we condition on was built from the statistics that *predict* era, "
        "so conditioning on it may hand the model the date through a side channel. Era directions "
        "are removed from the 14-dimensional space one at a time, **refitting a fresh classifier "
        "after each**, until it can no longer find the date. How many that takes is the measurement: "
        "a few means an era-free appearance vector exists, many means appearance and era are nearly "
        "the same thing here.",
        "",
        "| dimensions removed | era AUC | 95% CI | variance kept |",
        "|---|---|---|---|",
    ]
    lines += [f"| {int(r.dims_removed)} | {r.era_auc:.3f} | {r.ci_low:.2f}-{r.ci_high:.2f} | "
              f"{_pct(r.variance_kept)} |" for r in report["trace"].itertuples()]
    lines += [
        "", f"Stopping rule: era counts as removed below AUC {report['target_auc']:.2f}.",
        "", "### Which statistics carry the first era direction",
        "| features | share of the direction |", "|---|---|",
    ]
    lines += [f"| {r.features} | {_pct(r.share_of_era_direction)} |"
              for r in group_alignment(report["direction"]).itertuples()]
    top = report["loadings"].head(5)
    lines += ["", "Largest single loadings: " + ", ".join(f"`{r.feature}` ({r.era_loading:.2f})"
                                                          for r in top.itertuples()) + "."]
    if lighting:
        lines += [
            "", "### Does lighting survive?",
            f"*{lighting['n_positive']} night / {lighting['n_negative']} day labelled views in the "
            "test split.*", "",
            f"- From the full appearance vector, lighting is predicted at AUC **{lighting['auc_full']:.3f}**.",
            f"- With the era directions removed, **{lighting['auc_residual']:.3f}** "
            f"({_pct(lighting['retained'])} of the signal above chance retained).",
        ]
    else:
        lines += ["", "*No lighting labels supplied, so this only shows whether an era-free part of "
                      "appearance exists - not whether it still carries anything we want.*"]
    lines += ["", "### Verdict", verdict(report, lighting)]
    return "\n".join(lines) + "\n"
