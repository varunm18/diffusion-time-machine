"""CPU tests for appearance conditioning (no SEVA, no GPU).

The point of this module is factorization: the model should be able to express
"this photo is a brown, flat albumen print" without using the date to say it.
These tests pin down the properties that has to rest on - that a fresh module
changes nothing, that it nests under date FiLM instead of replacing it, that
appearance and date reach the network by separate paths, and that the features
the conditioning sees are the same ones we measured the shortcut with.
"""
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from timemachine.analysis import shortcut as sc
from timemachine.data.appearance import appearance_batch, appearance_from_tensor
from timemachine.model.appearance_film import (
    N_APPEARANCE_FEATURES, AppearanceEncoder, AppearanceFiLMConv, AppearanceState,
    appearance_film_layers, appearance_film_state_dict, load_appearance_film_state_dict,
)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL -> (3, H, W) in [-1, 1], the dataset's convention."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def noise_image(rng, size=64, brightness=0.5, contrast=0.2):
    lum = np.clip(rng.normal(brightness, contrast, (size, size)), 0, 1)
    return Image.fromarray((np.stack([lum] * 3, -1) * 255).astype(np.uint8), "RGB")


# ------------------------------------------------- features from model pixels
def test_tensor_features_match_the_measurement_code():
    """Conditioning and measurement must not drift apart: same numbers, same code."""
    rng = np.random.default_rng(0)
    image = sc.apply_sepia(noise_image(rng))
    direct = sc.feature_matrix([sc.appearance_features(image)])[0]
    through_tensor = appearance_from_tensor(to_tensor(image))
    assert through_tensor.shape == (N_APPEARANCE_FEATURES,)
    np.testing.assert_allclose(through_tensor, direct, rtol=1e-4, atol=1e-4)


def test_appearance_batch_separates_a_sepia_print_from_a_colour_photo():
    rng = np.random.default_rng(1)
    grey = noise_image(rng)
    colour = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), "RGB")
    images = torch.stack([to_tensor(sc.apply_sepia(grey)), to_tensor(colour), to_tensor(grey)])
    feats = appearance_batch(images)

    assert feats.shape == (3, N_APPEARANCE_FEATURES) and feats.dtype == torch.float32
    sat = sc.FEATURE_NAMES.index("saturation_mean")
    minor = sc.FEATURE_NAMES.index("chroma_minor")
    # A sepia print is strongly *tinted*, so plain saturation does not separate it
    # from colour - it can even be higher. What separates them is that a toned print
    # varies along a single colour axis, which is what chroma_minor measures.
    assert feats[1, minor] > 2 * feats[0, minor]
    assert feats[2, sat] < feats[0, sat]              # untoned grey is the least saturated
    assert torch.isfinite(feats).all()


# ------------------------------------------------------------------- encoder
def test_encoder_normalization_is_fitted_and_travels_with_the_checkpoint():
    rng = np.random.default_rng(2)
    stats = torch.from_numpy(rng.normal(5.0, 3.0, (200, N_APPEARANCE_FEATURES))).float()
    enc = AppearanceEncoder(dim=32).fit_normalization(stats)
    assert torch.allclose(enc.mean, stats.mean(0), atol=1e-4)
    assert torch.allclose(enc.scale, stats.std(0), atol=1e-4)
    assert "mean" in enc.state_dict() and "scale" in enc.state_dict()   # persistent buffers

    constant = stats.clone()
    constant[:, 3] = 7.0                                  # a statistic that never varies
    enc2 = AppearanceEncoder(dim=32).fit_normalization(constant)
    assert enc2.scale[3] == 1.0                           # must not divide by ~0
    assert torch.isfinite(enc2(constant[:4])).all()

    with pytest.raises(ValueError):
        AppearanceEncoder(dim=32).fit_normalization(torch.zeros(5))


def test_encoder_distinguishes_appearances_and_handles_unknown():
    rng = np.random.default_rng(3)
    stats = torch.from_numpy(rng.normal(0, 1, (64, N_APPEARANCE_FEATURES))).float()
    enc = AppearanceEncoder(dim=16).fit_normalization(stats)
    out = enc(stats[:4])
    assert out.shape == (4, 16)
    assert not torch.allclose(out[0], out[1])             # different looks, different features

    valid = torch.tensor([True, False, True, False])
    masked = enc(stats[:4], valid)
    assert torch.allclose(masked[0], out[0])              # kept rows unchanged
    assert not torch.allclose(masked[1], out[1])          # dropped rows replaced
    assert torch.allclose(masked[1], masked[3])           # ... all by the same learned vector


# ---------------------------------------------------------------- FiLM layer
def test_film_is_identity_at_init_and_responds_once_trained():
    state = AppearanceState()
    conv = nn.Conv2d(6, 4, 3, padding=1)
    layer = AppearanceFiLMConv(conv, state, feature_dim=16)
    h = torch.randn(3, 6, 8, 8)

    assert torch.equal(layer(h), conv(h))                 # no features set -> untouched
    state.features = torch.randn(3, 16)
    assert torch.allclose(layer(h), conv(h))              # zero init -> still exactly SEVA

    nn.init.normal_(layer.film.weight, std=0.1)
    assert not torch.allclose(layer(h), conv(h))          # now appearance matters
    # and different views get different modulation
    state.features = torch.zeros(3, 16)
    state.features[0] = 5.0
    out = layer(h)
    assert not torch.allclose(out[0], layer.inner(h)[0])


def test_film_rejects_a_view_count_mismatch():
    state = AppearanceState()
    layer = AppearanceFiLMConv(nn.Conv2d(6, 4, 3, padding=1), state, feature_dim=16)
    state.features = torch.randn(5, 16)
    with pytest.raises(ValueError, match="views vs"):
        layer(torch.randn(3, 6, 8, 8))


def test_film_nests_so_date_conditioning_keeps_working():
    """Appearance must wrap date FiLM, not replace it: both paths stay live."""
    class FakeDateFiLM(nn.Module):
        """Stand-in with date_film.DateFiLMConv's shape: a conv plus its own modulation."""

        def __init__(self, conv):
            super().__init__()
            self.conv = conv
            self.gain = nn.Parameter(torch.zeros(1))

        def forward(self, h):
            return self.conv(h * (1 + self.gain))

    conv = nn.Conv2d(6, 4, 3, padding=1)
    date = FakeDateFiLM(conv)
    state = AppearanceState()
    layer = AppearanceFiLMConv(date, state, feature_dim=16)

    assert layer.film.out_features == 2 * 6               # channel count found through the wrapper
    h = torch.randn(2, 6, 8, 8)
    assert torch.allclose(layer(h), date(h))              # zero init: unchanged

    with torch.no_grad():                                  # the inner (date) path still works
        date.gain.fill_(0.5)
    assert not torch.allclose(layer(h), conv(h))
    assert sum(p.numel() for p in layer.parameters()) > sum(p.numel() for p in date.parameters())


def test_film_needs_a_conv_somewhere_inside():
    with pytest.raises(ValueError, match="No Conv2d"):
        AppearanceFiLMConv(nn.Linear(4, 4), AppearanceState(), feature_dim=8)


# ----------------------------------------------------- saving and restoring
def test_state_dict_round_trip_over_a_small_stand_in_model():
    state = AppearanceState()
    model = nn.Sequential(
        AppearanceFiLMConv(nn.Conv2d(3, 4, 1), state, 8),
        AppearanceFiLMConv(nn.Conv2d(4, 4, 1), state, 8),
    )
    for layer in appearance_film_layers(model).values():
        nn.init.normal_(layer.film.weight, std=0.1)
        nn.init.normal_(layer.film.bias, std=0.1)

    saved = appearance_film_state_dict(model)
    assert len(appearance_film_layers(model)) == 2 and len(saved) == 4
    assert all(v.device.type == "cpu" for v in saved.values())

    fresh_state = AppearanceState()
    fresh = nn.Sequential(AppearanceFiLMConv(nn.Conv2d(3, 4, 1), fresh_state, 8),
                          AppearanceFiLMConv(nn.Conv2d(4, 4, 1), fresh_state, 8))
    load_appearance_film_state_dict(fresh, saved)
    for a, b in zip(appearance_film_layers(model).values(), appearance_film_layers(fresh).values()):
        assert torch.equal(a.film.weight, b.film.weight) and torch.equal(a.film.bias, b.film.bias)

    with pytest.raises(KeyError, match="state mismatch"):
        load_appearance_film_state_dict(fresh, {k: v for k, v in list(saved.items())[:1]})


def test_state_is_shared_so_one_write_reaches_every_block():
    """Gradient checkpointing recomputes blocks; they must all read the same tensor."""
    state = AppearanceState()
    layers = [AppearanceFiLMConv(nn.Conv2d(3, 3, 1), state, 8) for _ in range(3)]
    state.features = torch.randn(2, 8)
    assert all(layer._state is state for layer in layers)
    assert all(layer._state.features is state.features for layer in layers)
    # the state holder is not a submodule, so it never lands in a checkpoint
    assert not any("state" in name for name, _ in nn.Sequential(*layers).named_parameters())


# -------------------------------------------------------------- integration
def test_appearance_and_date_are_separate_pathways():
    """Changing the appearance vector must move the output without touching the date path."""
    state = AppearanceState()
    conv = nn.Conv2d(3, 3, 1)
    layer = AppearanceFiLMConv(conv, state, feature_dim=12)
    nn.init.normal_(layer.film.weight, std=0.2)
    h = torch.randn(2, 3, 4, 4)

    state.features = torch.zeros(2, 12)
    flat = layer(h)
    state.features = torch.ones(2, 12)
    toned = layer(h)
    assert not torch.allclose(flat, toned)
    # Nothing about the date was involved: the layer has only its own FiLM parameters.
    assert {n for n, _ in layer.named_parameters()} == {"film.weight", "film.bias",
                                                        "inner.weight", "inner.bias"}


def test_add_appearance_film_on_real_seva():
    seva = pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")
    from timemachine.model.appearance_film import add_appearance_film

    model = seva.Seva(model_channels=8, num_res_blocks=1)       # pragma: no cover - needs SEVA
    state = AppearanceState()
    names = add_appearance_film(model, state, feature_dim=16)
    assert names and len(appearance_film_layers(model)) == len(names)
    assert not add_appearance_film(model, state, 16)             # idempotent
