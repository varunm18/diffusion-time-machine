"""Era-residualized appearance (``--appearance-remove-era``): the encoder half.

Nothing here needs SEVA, so none of it can skip. The model-level wiring, which
does need SEVA's ResBlock, is in ``test_era_residual_seva.py``.

The arm this supports conditions on appearance with its era subspace projected
out, so the date pathway cannot read the year off film stock *and* the appearance
pathway cannot carry it either. What has to hold:

* k = 0 is exactly the encoder the other arms use;
* with k > 0, moving a view's statistics along the era subspace changes nothing
  the encoder produces, and a fresh classifier cannot find era in what it sees;
* a checkpoint loads whatever its k, including checkpoints from before the buffer.
"""
import numpy as np
import pytest
import torch

from timemachine.analysis.shortcut import YearProbe, balanced_weights, roc_auc
from timemachine.data.appearance import dataset_appearance_stats
from timemachine.model.appearance_film import (
    N_APPEARANCE_FEATURES, AppearanceEncoder, fit_era_directions,
)

F = N_APPEARANCE_FEATURES


def era_data(n=1200, seed=0):
    """Raw statistics on very different scales, with era shifting two correlated axes.

    The noise is correlated, so the best linear era direction is not the mean
    shift itself; removing it leaves some era behind and the fit has to go again.
    That exercises k > 1, which is what the real data needed (6 of 14).
    """
    rng = np.random.default_rng(seed)
    years = np.where(rng.random(n) < 0.3, rng.uniform(1860, 1940, n), rng.uniform(1950, 2020, n))
    old = years < 1945
    mix = np.eye(F) + 0.6 * rng.normal(size=(F, F)) / np.sqrt(F)
    z = rng.normal(size=(n, F)) @ mix
    z[:, 0] += 1.5 * old
    z[:, 3] += 1.0 * old
    raw_scale = np.geomspace(0.01, 50.0, F)              # saturation next to Lab chroma spread
    raw = 3.0 + z * raw_scale
    return torch.from_numpy(raw).float(), years


def fitted_encoder(seed=0):
    stats, years = era_data(seed=seed)
    enc = AppearanceEncoder(dim=16).fit_normalization(stats)
    report = fit_era_directions(enc, stats, years, seed=seed)
    return enc, stats, years, report


def mlp_input(enc, stats, valid=None):
    """What the encoder's MLP actually receives, captured from the real forward pass."""
    seen = {}
    handle = enc.mlp.register_forward_pre_hook(lambda _m, args: seen.setdefault("z", args[0].detach()))
    try:
        enc(stats, valid)
    finally:
        handle.remove()
    return seen["z"]


def held_out_era_auc(z, years, seed=1):
    rng = np.random.default_rng(seed)
    train = rng.random(len(years)) < 0.6
    old = years < 1945
    probe = YearProbe().fit(z[train], np.where(old[train], 1.0, -1.0), balanced_weights(years[train]))
    return roc_auc(probe.predict(z[~train]), old[~train])


# ------------------------------------------------------------------ k = 0

def test_a_fresh_encoder_removes_nothing():
    enc = AppearanceEncoder(dim=16)
    assert not enc.removes_era
    assert tuple(enc.era_directions.shape) == (0, F)
    assert "era_directions" in enc.state_dict()          # persistent: travels with checkpoints


def test_zero_directions_is_exactly_the_plain_encoder():
    torch.manual_seed(0)
    stats, _ = era_data(n=40)
    enc = AppearanceEncoder(dim=16).fit_normalization(stats)
    before = enc(stats)
    enc.set_era_directions(np.zeros((0, F)))
    assert not enc.removes_era
    assert torch.equal(enc(stats), before)
    enc.set_era_directions([])                            # any empty input means "off"
    assert torch.equal(enc(stats), before)


def test_turning_removal_off_again_restores_the_plain_encoder():
    enc, stats, _, _ = fitted_encoder()
    removed = enc(stats[:8])
    enc.set_era_directions(torch.zeros(0, F))
    plain = enc(stats[:8])
    assert not torch.allclose(removed, plain)
    enc.set_era_directions(torch.zeros(0, F))
    assert torch.equal(enc(stats[:8]), plain)


# ------------------------------------------------------------------ the projection

def test_directions_are_stored_orthonormal():
    enc = AppearanceEncoder(dim=16)
    raw = torch.randn(3, F)
    enc.set_era_directions(raw)
    d = enc.era_directions
    torch.testing.assert_close(d @ d.T, torch.eye(3), atol=1e-5, rtol=0)
    # Same subspace as what was given: projecting the inputs onto it loses nothing.
    torch.testing.assert_close((raw @ d.T) @ d, raw, atol=1e-5, rtol=1e-5)


def test_dependent_or_misshapen_directions_are_refused():
    enc = AppearanceEncoder(dim=16)
    d = torch.randn(1, F)
    with pytest.raises(ValueError, match="dependent"):
        enc.set_era_directions(torch.cat([d, 2 * d]))
    with pytest.raises(ValueError):
        enc.set_era_directions(torch.randn(2, F - 1))
    with pytest.raises(ValueError):
        enc.set_era_directions(torch.randn(F + 1, F))


def test_the_fit_finds_era_and_removes_it():
    enc, stats, years, report = fitted_encoder()
    assert enc.removes_era
    assert report["dims_removed"] == enc.era_directions.shape[0] >= 1
    assert report["era_auc_full"] > 0.8                    # the synthetic shortcut is real
    assert report["era_removed"] and report["era_auc_residual"] < 0.60

    z = mlp_input(enc, stats)
    # Nothing left along the removed directions ...
    assert (z @ enc.era_directions.T).abs().max() < 1e-4
    # ... and a classifier that was never told which directions were removed cannot
    # find era in what the encoder actually sees (rows reshuffled vs. the fit's split).
    assert held_out_era_auc(z.double().numpy(), years) < 0.62
    # While the unprojected input still carries it, so the test is not vacuous.
    plain = (stats - enc.mean) / enc.scale
    assert held_out_era_auc(plain.double().numpy(), years) > 0.8


def test_the_fit_needs_more_than_one_direction_on_correlated_data():
    """One removal is rarely enough (findings: 6 of 14); the fit must iterate."""
    _, _, _, report = fitted_encoder()
    assert report["dims_removed"] >= 2, report["trace"]


def test_moving_along_the_era_subspace_changes_nothing_downstream():
    enc, stats, _, _ = fitted_encoder()
    torch.manual_seed(3)
    coeffs = torch.randn(len(stats), enc.era_directions.shape[0]) * 3.0
    # A step along the era subspace in standardized space, mapped back to raw statistics.
    shifted = stats + (coeffs @ enc.era_directions) * enc.scale
    torch.testing.assert_close(enc(shifted), enc(stats), atol=1e-5, rtol=1e-4)
    # Whereas a step orthogonal to it is still seen: appearance is not erased wholesale.
    d = enc.era_directions
    other = torch.randn(F)
    other = other - (other @ d.T) @ d
    moved = stats + 3.0 * (other / other.norm()) * enc.scale
    assert not torch.allclose(enc(moved), enc(stats), atol=1e-3)


def test_projection_happens_after_standardization():
    """In the raw basis the same subtraction would mix a 0.01-scale statistic with a
    50-scale one; the era step must vanish in the basis the directions were fitted in."""
    enc, stats, _, _ = fitted_encoder()
    d = enc.era_directions[0]
    raw_step = stats + 2.0 * d                           # NOT rescaled: a step in raw units
    # A raw-unit step along d is not an era step for the encoder, so it shows up.
    assert not torch.allclose(enc(raw_step), enc(stats), atol=1e-4)


def test_unknown_appearance_still_replaces_the_projected_vector():
    enc, stats, _, _ = fitted_encoder()
    valid = torch.tensor([True, False, True, False])
    z = mlp_input(enc, stats[:4], valid)
    torch.testing.assert_close(z[1], enc.unknown.detach())
    torch.testing.assert_close(z[3], enc.unknown.detach())


def test_the_fit_refuses_data_with_only_one_era():
    stats, years = era_data(n=100)
    enc = AppearanceEncoder(dim=16).fit_normalization(stats)
    with pytest.raises(ValueError, match="old and modern"):
        fit_era_directions(enc, stats, np.full(len(stats), 2000.0))
    assert not enc.removes_era                           # nothing half-installed


def test_rows_without_a_year_are_left_out_of_the_fit():
    stats, years = era_data()
    years = years.copy()
    years[:50] = np.nan
    enc = AppearanceEncoder(dim=16).fit_normalization(stats)
    report = fit_era_directions(enc, stats, years)
    assert report["n_rows"] == len(stats) - 50
    assert enc.removes_era


def test_the_report_is_json_ready():
    import json

    _, _, _, report = fitted_encoder()
    json.dumps(report)                                   # the trainer writes it next to the run


# ------------------------------------------------------------------ checkpoints

@pytest.mark.parametrize("saved_k,fresh_k", [(3, 0), (0, 3), (2, 5), (4, 4)])
def test_a_checkpoint_with_a_different_k_loads(saved_k, fresh_k):
    """A fresh module is built with k = 0 before any checkpoint says otherwise; a size
    mismatch here broke every resume and evaluation the last time this was built."""
    torch.manual_seed(saved_k)
    saved = AppearanceEncoder(dim=16).fit_normalization(torch.randn(50, F) * 4 + 1)
    saved.set_era_directions(torch.randn(saved_k, F))
    fresh = AppearanceEncoder(dim=16)
    fresh.set_era_directions(torch.randn(fresh_k, F))

    result = fresh.load_state_dict(saved.state_dict())    # strict
    assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(fresh.era_directions, saved.era_directions)
    assert fresh.removes_era == (saved_k > 0)
    probe = torch.randn(6, F) * 4 + 1
    assert torch.equal(fresh(probe), saved(probe))


def test_a_checkpoint_from_before_the_buffer_loads_as_k_zero():
    """Arm-2 checkpoints have mean/scale but no era_directions key."""
    saved = AppearanceEncoder(dim=16).fit_normalization(torch.randn(50, F))
    state = saved.state_dict()
    del state["era_directions"]
    fresh = AppearanceEncoder(dim=16).set_era_directions(torch.randn(2, F))
    result = fresh.load_state_dict(state)                 # strict, and must not complain
    assert not result.missing_keys
    assert not fresh.removes_era
    probe = torch.randn(6, F)
    assert torch.equal(fresh(probe), saved(probe))


def test_a_wrong_width_checkpoint_still_fails_loudly():
    """The resize is for k only; a checkpoint for a different feature set is a real error."""
    saved = AppearanceEncoder(n_features=F - 1, dim=16)
    saved.set_era_directions(torch.randn(2, F - 1))
    with pytest.raises(RuntimeError):
        AppearanceEncoder(dim=16).load_state_dict(saved.state_dict())


def test_a_loaded_buffer_lives_where_the_module_does():
    saved = AppearanceEncoder(dim=16).set_era_directions(torch.randn(2, F))
    fresh = AppearanceEncoder(dim=16).double()
    fresh.load_state_dict(saved.state_dict())
    assert fresh.era_directions.dtype == fresh.mean.dtype


# ------------------------------------------------------------------ the statistics

class FakeDataset:
    """Samples that share views, as overlapping real samples do."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.views = {f"v{i}": (torch.from_numpy(rng.normal(size=F)).float(), 1900.0 + 10 * i)
                      for i in range(10)}
        self.samples = [["v0", "v1", "v2"], ["v2", "v3", "v4"], ["v4", "v5", "v0"], ["v6", "v7", "v8"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        keys = self.samples[i]
        return {"appearance": torch.stack([self.views[k][0] for k in keys]),
                "date_mid": torch.tensor([self.views[k][1] for k in keys], dtype=torch.float32),
                "file_keys": keys}


def test_stats_without_dates_are_unchanged():
    ds = FakeDataset()
    stats = dataset_appearance_stats(ds, n_samples=4)
    assert isinstance(stats, torch.Tensor) and stats.shape == (12, F)   # duplicates kept, as before


def test_stats_with_dates_pair_each_view_with_its_year_once():
    ds = FakeDataset()
    stats, years = dataset_appearance_stats(ds, n_samples=4, with_dates=True)
    assert stats.shape == (9, F) and years.shape == (9,)                # v0, v2, v4 once each
    by_year = {float(y): s for s, y in zip(stats, years)}
    for key, (feat, year) in ds.views.items():
        if key != "v9":                                                 # never drawn
            assert torch.equal(by_year[year], feat), key


def test_drawing_stats_does_not_keep_the_images_alive():
    """A real sample is ~30 MiB of pixels; holding 256 of them pinned ~7.6 GiB.

    What matters is the peak *during* the draw - once the function returns, a list
    of whole samples is freed anyway, so checking afterwards would pass the bug. So
    each time a sample is requested, count how many earlier samples' images are
    still alive: it must be none.
    """
    import weakref

    class WithImages(FakeDataset):
        def __init__(self):
            super().__init__()
            self.refs, self.alive_at_request = [], []

        def __getitem__(self, i):
            self.alive_at_request.append(sum(ref() is not None for ref in self.refs))
            sample = super().__getitem__(i)
            sample["images"] = torch.zeros(3, 3, 64, 64)
            self.refs.append(weakref.ref(sample["images"]))
            return sample

    ds = WithImages()
    stats, years = dataset_appearance_stats(ds, n_samples=4, with_dates=True)
    assert len(ds.refs) == 4
    assert ds.alive_at_request == [0, 0, 0, 0], "earlier samples' images were still held"
    assert stats.shape[0] == years.shape[0] == 9
