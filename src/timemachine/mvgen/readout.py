"""A colour-blind read-out of the building's state in a picture: before or after Notre-Dame's fire.

Experiment look1's read-outs R2 and R4 (``timemachine.mvgen.evaluation``) ask which state of the
building a render shows, and the answer must not move with the render's look: a look code that only
recolours a picture must not count as changing the state. The instrument is the appearance study's
(``data/appearance_study/design_learned/04a_readout_features.py``, ``04b_readout_probe.py``):

* **Crop** (:func:`region_crop`, :func:`readout_crop`): the picture at 256 x 256 (area average), the
  bounding box of the changed region grown to 1.5 times its longer side (at least 48, at most 256
  pixels), made square and kept inside the frame; then grey (Rec. 709 luminance), brightness and
  contrast normalised (``(g - mean) / (sd + 0.001) * 0.2 + 0.5``), resized to 224 (bicubic) and
  repeated into three channels. A global change of colour, brightness or contrast cannot move it.
* **Encoder** (:class:`DinoV2`): frozen DINOv2-base (``facebook/dinov2-base``, ImageNet normalisation),
  whose pooled output (the class token after the last layer norm) is the feature. MVGenMaster's
  environment has transformers 4.30, which predates DINOv2, so the network is written out here and
  loads the cached weights itself; it computes what ``transformers.Dinov2Model`` 5.x computes (the
  position embeddings resized bicubically to the patch grid).
* **Probe** (:class:`StateProbe`): logistic regression on the L2-normalised, standardised features
  (``C = 0.05``), fitted on real photos: 1 after the fire, 0 before it. Its logit is the read-out:
  above 0 the picture shows the building after the fire.

Measured in the study on 564 real photos of modern and after-fire views whose region covers 3% of
the frame or more: area under the ROC curve 0.998 with folds by capture day; recolouring a photo to
the statistics of a photo of the other period moves the logit by 0.23 +- 0.05, 2% of the gap between
the periods (10.5 logits; sd 2.25 inside a period). It does move with the season inside the after-fire
period (spring minus other about -1 logit), which is why R4's donors are matched on season.

:class:`StateReadout` is the fitted instrument: encoder and probe together, with
:meth:`StateReadout.readout_logit` for renders and photos. ``scripts/mvgen/fit_readout.py`` fits the probe on
the training views of experiment look1's hold-out (never on held-out views) and writes it with
:func:`save_probe`; the file names the views it was fitted on, so that an evaluation can check that it
never reads a view it was fitted on.

OpenCV (the resizing, as in the study) is needed for the crops and scikit-learn for fitting;
MVGenMaster's environment has both. Nothing here imports MVGenMaster's code.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DINOV2_REPO = "facebook/dinov2-base"
WORK_SIZE = 256                     # pictures are reduced to this side before cropping, photos and renders alike
CROP_SIZE = 224                     # the encoder's input side
CROP_GROWTH = 1.5                   # the crop's side over the region's longer bounding-box side
CROP_MIN, CROP_MAX = 48, WORK_SIZE  # bounds of the crop's side, at WORK_SIZE
GREY = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)        # Rec. 709 luminance, as the study
CONTRAST_SD, CONTRAST_MEAN, CONTRAST_EPS = 0.2, 0.5, 1e-3
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PROBE_C = 0.05                      # inverse regularisation of the logistic regression, as the study
MIN_REGION_SHARE = 0.03             # a picture is read only if its region covers this share of the frame
BATCH = 16


# ----------------------------------------------------------------------------
# Crops
# ----------------------------------------------------------------------------
def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError("the read-out's crops need OpenCV (cv2), which MVGenMaster's environment has") from error
    return cv2


def region_crop(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """uint8 ``[224, 224, 3]``: the grey, contrast-normalised crop around ``region`` of a 256 x 256 picture.

    ``image`` uint8 ``[256, 256, 3]``, ``region`` bool ``[256, 256]`` with at least one pixel. As the
    study's ``region_crop`` (module docstring).
    """
    cv2 = _cv2()
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        raise ValueError("a read-out crop needs a region with at least one pixel")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(CROP_MIN, min(int(max(y1 - y0, x1 - x0) * CROP_GROWTH), CROP_MAX))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, WORK_SIZE - side))
    left = int(np.clip(cx - side // 2, 0, WORK_SIZE - side))
    crop = image[top:top + side, left:left + side].astype(np.float32) / 255
    grey = crop @ GREY
    grey = (grey - grey.mean()) / (grey.std() + CONTRAST_EPS) * CONTRAST_SD + CONTRAST_MEAN
    grey = cv2.resize(np.clip(grey, 0, 1), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
    return (np.clip(np.repeat(grey[..., None], 3, -1), 0, 1) * 255).round().astype(np.uint8)


def readout_crop(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """:func:`region_crop` of a picture of any square size (uint8 ``[S, S, 3]``, ``region`` bool ``[S, S]``).

    The picture is first reduced to 256 x 256 by area averaging and the region by nearest neighbours,
    as the study treated its photos and renders (a region made of whole 8 x 8 cells at 512 gives the
    study's crops of real photos exactly).
    """
    cv2 = _cv2()
    image, region = np.asarray(image), np.asarray(region, dtype=bool)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != image.shape[1] or image.shape[2] != 3:
        raise ValueError(f"a read-out picture is uint8 [S, S, 3]; got {image.dtype} {image.shape}")
    if region.shape != image.shape[:2]:
        raise ValueError(f"the region {region.shape} is not the picture's size {image.shape[:2]}")
    if image.shape[0] != WORK_SIZE:
        image = cv2.resize(image, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
        region = cv2.resize(region.astype(np.uint8), (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_NEAREST) > 0
    return region_crop(image, region)


# ----------------------------------------------------------------------------
# The encoder: DINOv2, written out (its parameter names are those of the released weights)
# ----------------------------------------------------------------------------
class _SelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query, self.key, self.value = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)


class _AttentionOutput(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.dense = nn.Linear(width, width)


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attention, self.output = _SelfAttention(width), _AttentionOutput(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        q, k, v = (layer(x).view(batch, tokens, self.heads, -1).transpose(1, 2)
                   for layer in (self.attention.query, self.attention.key, self.attention.value))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.output.dense(out.transpose(1, 2).reshape(batch, tokens, width))


class _LayerScale(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class _Mlp(nn.Module):
    def __init__(self, width: int, ratio: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, width * ratio), nn.Linear(width * ratio, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Layer(nn.Module):
    def __init__(self, width: int, heads: int, ratio: int, eps: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width, eps=eps), nn.LayerNorm(width, eps=eps)
        self.attention, self.mlp = _Attention(width, heads), _Mlp(width, ratio)
        self.layer_scale1, self.layer_scale2 = _LayerScale(width), _LayerScale(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class _PatchEmbeddings(nn.Module):
    def __init__(self, width: int, patch: int):
        super().__init__()
        self.projection = nn.Conv2d(3, width, kernel_size=patch, stride=patch)


class _Embeddings(nn.Module):
    def __init__(self, width: int, patch: int, positions: int):
        super().__init__()
        self.patch = patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, width))         # unused: no masked patches
        self.patch_embeddings = _PatchEmbeddings(width, patch)
        self.position_embeddings = nn.Parameter(torch.zeros(1, positions + 1, width))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = pixels.shape[-2] // self.patch, pixels.shape[-1] // self.patch
        patches = self.patch_embeddings.projection(pixels).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(len(pixels), -1, -1), patches], dim=1)
        position = self.position_embeddings
        if patches.shape[1] != position.shape[1] - 1:
            side, dim = int(round((position.shape[1] - 1) ** 0.5)), position.shape[-1]
            grid = position[:, 1:].reshape(1, side, side, dim).permute(0, 3, 1, 2).float()
            grid = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False).to(position.dtype)
            position = torch.cat([position[:, :1], grid.permute(0, 2, 3, 1).reshape(1, -1, dim)], dim=1)
        return tokens + position


class _Encoder(nn.Module):
    def __init__(self, depth: int, width: int, heads: int, ratio: int, eps: float):
        super().__init__()
        self.layer = nn.ModuleList(_Layer(width, heads, ratio, eps) for _ in range(depth))


class DinoV2(nn.Module):
    """DINOv2's vision transformer (no SwiGLU, no registers), as ``transformers.Dinov2Model`` computes it.

    Called on ImageNet-normalised pixels ``[B, 3, H, W]`` (H, W multiples of the patch size), it returns
    the pooled output ``[B, width]``: the class token after the last layer norm.
    """

    def __init__(self, width: int = 768, depth: int = 12, heads: int = 12, patch: int = 14, positions: int = 37 ** 2,
                 mlp_ratio: int = 4, eps: float = 1e-6):
        super().__init__()
        self.embeddings = _Embeddings(width, patch, positions)
        self.encoder = _Encoder(depth, width, heads, mlp_ratio, eps)
        self.layernorm = nn.LayerNorm(width, eps=eps)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixels)
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)[:, 0]

    @classmethod
    def from_folder(cls, folder: str | Path) -> DinoV2:
        """The network with the weights of a downloaded model folder (``config.json``, ``model.safetensors``)."""
        from safetensors.torch import load_file

        folder = Path(folder)
        config = json.loads((folder / "config.json").read_text())
        if config.get("hidden_act") != "gelu" or config.get("use_swiglu_ffn") or config.get("num_register_tokens"):
            raise ValueError(f"{folder} is not a DINOv2 this module writes out (GELU, no SwiGLU, no registers)")
        patch = config["patch_size"]
        model = cls(width=config["hidden_size"], depth=config["num_hidden_layers"], heads=config["num_attention_heads"],
                    patch=patch, positions=(config["image_size"] // patch) ** 2, mlp_ratio=config["mlp_ratio"],
                    eps=config["layer_norm_eps"])
        model.load_state_dict(load_file(str(folder / "model.safetensors")))
        return model.eval().requires_grad_(False)


def dinov2_folder(repo: str = DINOV2_REPO) -> Path:
    """The folder of the cached DINOv2 weights (``HF_HOME``'s hub cache; nothing is downloaded).

    ``FileNotFoundError`` if the model is not in the cache.
    """
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo, "model.safetensors")
    if not isinstance(found, str):
        raise FileNotFoundError(f"{repo} is not in the Hugging Face cache (set HF_HOME to the cache that holds it)")
    return Path(found).parent


def encode(encoder: nn.Module, crops: np.ndarray, device: str | torch.device = "cpu", batch: int = BATCH
           ) -> np.ndarray:
    """float32 ``[n, width]``: the encoder's features of read-out crops (uint8 ``[n, 224, 224, 3]``)."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            pixels = torch.from_numpy(np.ascontiguousarray(crops[start:start + batch])).to(device)
            pixels = pixels.permute(0, 3, 1, 2).float() / 255
            out.append(encoder((pixels - mean) / std).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.layernorm.normalized_shape[0]), np.float32)


# ----------------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StateProbe:
    """Logistic regression on L2-normalised, standardised features: what ``make_pipeline(Normalizer(),
    StandardScaler(), LogisticRegression(C=0.05))`` of the study computes, as plain arrays."""

    mean: np.ndarray            # [width], the StandardScaler's mean of the normalised features
    scale: np.ndarray           # [width], its scale (1 where a feature has no spread)
    coef: np.ndarray            # [width]
    intercept: float

    def logit(self, features: np.ndarray) -> np.ndarray:
        """The decision function (log odds of "after the fire") of features ``[n, width]``."""
        features = np.asarray(features, dtype=np.float64)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalised = features / np.where(norms > 0, norms, 1.0)
        return ((normalised - self.mean) / self.scale) @ self.coef + self.intercept


def fit_probe(features: np.ndarray, after: np.ndarray, c: float = PROBE_C) -> StateProbe:
    """The study's probe fitted on features ``[n, width]`` and labels ``after`` (bool ``[n]``: after the fire)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler

    pipeline = make_pipeline(Normalizer(), StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    pipeline.fit(np.asarray(features, dtype=np.float64), np.asarray(after, dtype=int))
    scaler, regression = pipeline[1], pipeline[2]
    return StateProbe(mean=scaler.mean_.copy(), scale=scaler.scale_.copy(), coef=regression.coef_[0].copy(),
                      intercept=float(regression.intercept_[0]))


def save_probe(path: str | Path, probe: StateProbe, record: Mapping[str, object]) -> Path:
    """Write a probe and its ``record`` (JSON types: what it was fitted on and how; ``fitted_on`` must list the
    file keys of the views it was fitted on) to an ``.npz`` file, under another name and then renamed."""
    if "fitted_on" not in record:
        raise ValueError("a read-out's record must name the views it was fitted on (fitted_on)")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as file:
        np.savez(file, mean=probe.mean, scale=probe.scale, coef=probe.coef, intercept=np.float64(probe.intercept),
                 record=json.dumps(dict(record)))
    tmp.replace(path)
    return path


def load_probe(path: str | Path) -> tuple[StateProbe, dict]:
    """The probe and its record written by :func:`save_probe`."""
    with np.load(path) as stored:
        probe = StateProbe(mean=stored["mean"], scale=stored["scale"], coef=stored["coef"],
                           intercept=float(stored["intercept"]))
        return probe, json.loads(str(stored["record"]))


# ----------------------------------------------------------------------------
# The instrument
# ----------------------------------------------------------------------------
class StateReadout:
    """The fitted read-out: encoder and probe. ``readout_logit(images, regions)`` reads pictures.

    Args:
        probe, record: from :func:`load_probe`.
        encoder: the frozen :class:`DinoV2` (by default the cached ``facebook/dinov2-base``).
        device: where the encoder runs.
    """

    def __init__(self, probe: StateProbe, record: Mapping[str, object], encoder: nn.Module | None = None,
                 device: str | torch.device = "cpu"):
        self.probe, self.record, self.device = probe, dict(record), device
        self.encoder = (encoder if encoder is not None else DinoV2.from_folder(dinov2_folder())).to(device).eval()
        self.min_share = float(self.record.get("min_region_share", MIN_REGION_SHARE))
        self.fitted_on = frozenset(self.record["fitted_on"])

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu", encoder: nn.Module | None = None
             ) -> StateReadout:
        probe, record = load_probe(path)
        return cls(probe, record, encoder, device)

    def readable(self, regions: np.ndarray) -> np.ndarray:
        """bool ``[n]``: which of the regions (bool ``[n, S, S]``) cover enough of the frame to be read."""
        regions = np.asarray(regions, dtype=bool)
        return regions.reshape(len(regions), -1).mean(1) >= self.min_share

    def readout_logit(self, images: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """float ``[n]``: the read-out of pictures (uint8 ``[n, S, S, 3]``) around their changed regions (bool
        ``[n, S, S]``); NaN where a region covers less than ``min_share`` of the frame. Above 0: after the fire."""
        images, regions = np.asarray(images), np.asarray(regions, dtype=bool)
        out = np.full(len(images), np.nan)
        read = np.flatnonzero(self.readable(regions))
        if len(read):
            crops = np.stack([readout_crop(images[i], regions[i]) for i in read])
            out[read] = self.probe.logit(encode(self.encoder, crops, self.device))
        return out

    def check_unseen(self, file_keys: Sequence[str]) -> None:
        """``ValueError`` if the probe was fitted on any of these views: it would read photos it has learnt."""
        seen = sorted(set(file_keys) & self.fitted_on)
        if seen:
            raise ValueError(f"the state read-out was fitted on {len(seen)} of the views it is asked to score "
                             f"(e.g. {seen[:3]}): fit it on the training views of this hold-out")
