"""Hold-out of views from training, and the held-out vs trained probe."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from timemachine.data.dataset import DatedViewsDataset
from timemachine.data.holdout import HoldoutConfig, era_index, heldout_rows
from timemachine.data.sampling import SamplingConfig, eligible_anchors, sample_views, tolerance

# 20 old views in two date clusters, 40 modern views in two clusters.
DATES = np.concatenate([np.full(10, 1890.0), np.full(10, 1935.0), np.full(20, 2012.0), np.full(20, 2020.0)])
CFG = SamplingConfig(n_inputs=3, n_targets=5)


@pytest.fixture
def views(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i, date in enumerate(DATES):
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(path)
        angle = i * 0.1
        rows.append(dict(
            scene_id=1, model=0, file_key=f"{i}.jpg", image_path=str(path), image_width=32, image_height=24,
            camera_width=320, camera_height=240, fx=300.0, fy=300.0, cx=160.0, cy=120.0, k1=0.0, k2=0.0,
            qw=np.cos(angle / 2), qx=0.0, qy=np.sin(angle / 2), qz=0.0, tx=0.05 * i, ty=0.0, tz=3.0,
            date_lo=date, date_hi=date + 1, date_mid=date, medium="photo", is_monochrome=date < 1970))
    return pd.DataFrame(rows)


def test_heldout_rows_are_stratified_by_era_and_include_extra_rows():
    cfg = HoldoutConfig(fraction=0.1, seed=3)
    rows = heldout_rows(DATES, cfg, extra_rows=[0, 1])
    era = era_index(DATES[rows], cfg.era_edges)
    assert {0, 1} <= set(rows)
    assert (era == 0).sum() >= 1 and (era == 1).sum() == 1 and (era == 3).sum() == 4   # 10%: 1, 1, 4 (+2 extra)
    assert np.array_equal(rows, heldout_rows(DATES, cfg, extra_rows=[0, 1]))
    assert len(heldout_rows(DATES, HoldoutConfig())) == 0
    with pytest.raises(ValueError):
        HoldoutConfig(fraction=1.0)


def test_hold_out_keeps_rows_out_of_samples_and_nothing_held_out_is_the_old_stream(views):
    ds = DatedViewsDataset(views, CFG, resolution=16)
    anchors = eligible_anchors(DATES, CFG)
    for i in range(10):                                  # no hold-out: identical to sampling the full table
        rng = np.random.default_rng((ds.seed, ds.epoch, i))
        assert np.array_equal(ds.indices(i), np.concatenate(sample_views(DATES, CFG, rng, anchors, ds.camera_feats)))
    held = np.array([0, 5, 12, 25, 33, 41, 59])
    ds.hold_out(held)
    assert ds.heldout.sum() == len(held) and len(ds.train_rows) == len(DATES) - len(held)
    for epoch in range(3):
        for i in range(30):
            assert not np.isin(ds.indices(i, epoch=epoch), held).any()
    assert ds.make_sample(np.array([1, 2, 3, 0, 4, 6, 7, 8]), 3)["file_keys"][3] == "0.jpg"   # still loadable


def test_measured_view_rows_and_pairs_use_trained_views_only(views):
    pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")
    from timemachine.training.probes import heldout_pairs, measured_view_rows
    ds = DatedViewsDataset(views, CFG, resolution=16)
    ds.hold_out(np.array([2, 13, 30, 50]))
    rows = measured_view_rows(ds, 2, ds.train_rows)
    assert len(rows) == 8 and rows[3] == 2
    others = np.delete(rows, 3)
    assert ds.heldout[others].sum() == 0
    assert np.all(np.abs(DATES[rows[4:]] - DATES[2]) <= tolerance(CFG, DATES[2]))
    small = ds.train_rows[DATES[ds.train_rows] > 2000]           # no same-date siblings for an 1890 view
    assert measured_view_rows(ds, 2, small) is None

    pairs = heldout_pairs(ds, 10, np.random.default_rng(0), (1900.0, 1970.0, 2000.0))
    assert len(pairs) == 4 and {k for *_, k in pairs} == {"view"}
    assert [DATES[h[3]] < 1970 for h, _, _ in pairs] == [True, True, False, False]   # old ones first
    for h, t, _ in pairs:
        assert ds.heldout[h[3]] and not ds.heldout[t[3]]
        assert era_index(DATES[[h[3]]], (1900.0, 1970.0, 2000.0)) == era_index(DATES[[t[3]]], (1900.0, 1970.0, 2000.0))
        assert ds.heldout[np.delete(h, 3)].sum() == 0 and ds.heldout[t].sum() == 0


class ScaleModel(nn.Module):
    """eps_hat = a * x: a stand-in denoiser whose quality is set by ``a``."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t, cond, num_frames, date=None, date_valid=None):
        return self.a * x


def test_heldout_probe_reports_change_from_its_first_evaluation(views, tmp_path, monkeypatch):
    pytest.importorskip("seva.eval", reason="SEVA not installed (scripts/setup_seva.sh)")
    from timemachine.training import probes
    from timemachine.training.conditioning import build_conditions

    def fake_conditions(components, sample, device, camera_scale):
        V = len(sample["is_input"])
        g = torch.Generator().manual_seed(int(sample["date_mid"].sum()))
        latents = torch.randn(V, 4, 2, 2, generator=g)
        return (latents, *build_conditions(latents, torch.randn(1024, generator=g),
                                           torch.randn(V, 6, 2, 2, generator=g), sample["is_input"]))

    monkeypatch.setattr(probes, "sample_conditions", fake_conditions)
    no_autocast = torch.autocast
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: no_autocast("cpu", enabled=False))
    ds = DatedViewsDataset(views, CFG, resolution=16)
    ds.hold_out(np.array([2, 13, 30, 50]))
    probe = probes.HeldoutProbe(None, ds, tmp_path / "base.json", n_pairs=4, device="cpu", date_views="targets")
    sigmas = torch.linspace(0.03, 80, probes.NUM_LEVELS)
    model = ScaleModel()
    first = probe(model, sigmas)
    assert first["pairs"] == 4 and first["old_pairs"] == 2
    assert first["heldout_change"] == pytest.approx(0.0) and first["memorisation_gap"] == pytest.approx(0.0)
    assert first["heldout_rel_unknown"] == pytest.approx(0.0)        # this model ignores dates
    model.a.data.fill_(0.9)
    later = probes.HeldoutProbe(None, ds, tmp_path / "base.json", n_pairs=4, device="cpu")(model, sigmas)
    assert later["heldout_change"] != pytest.approx(0.0)              # compared with the stored baseline


# --- several scenes ---------------------------------------------------------------
@pytest.fixture
def two_scenes(views):
    return pd.concat([views, views.assign(scene_id=2, file_key="b" + views["file_key"])], ignore_index=True)


def test_heldout_rows_per_scene_reduce_to_era_strata_for_one_scene():
    cfg = HoldoutConfig(fraction=0.2, seed=1)
    assert np.array_equal(heldout_rows(DATES, cfg), heldout_rows(DATES, cfg, groups=np.zeros(len(DATES), int)))
    groups = np.repeat([0, 1], len(DATES))
    rows = heldout_rows(np.tile(DATES, 2), cfg, groups=groups)
    assert (groups[rows] == 0).sum() == (groups[rows] == 1).sum() == len(heldout_rows(DATES, cfg))


def test_pairs_cover_heldout_views_and_heldout_scenes(two_scenes):
    pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")
    from timemachine.training.probes import heldout_pairs
    ds = DatedViewsDataset(two_scenes, CFG, resolution=16)
    scene2 = ds.group_rows[1]
    ds.hold_out(np.concatenate([[2, 30], scene2]))
    pairs = heldout_pairs(ds, 6, np.random.default_rng(0), (1900.0, 1970.0, 2000.0))
    kinds = [k for *_, k in pairs]
    assert kinds[:4] == ["view", "scene", "view", "scene"] and len(pairs) == 6        # kinds alternate
    for h, t, kind in pairs:
        assert len(set(ds.group[h])) == 1 and len(set(ds.group[t])) == 1           # one frame per sample
        assert not ds.heldout[t].any()
        if kind == "scene":
            assert ds.group[h[0]] == 1 and ds.heldout[h].all()                     # built from unseen views
        else:
            assert ds.heldout[h].sum() == 1


def test_render_probe_rows_stay_in_their_scene(two_scenes):
    pytest.importorskip("seva.model", reason="SEVA not installed (scripts/setup_seva.sh)")
    from timemachine.training.probes import RenderProbeSpec, select_probe_rows
    ds = DatedViewsDataset(two_scenes, CFG, resolution=16)
    spec = RenderProbeSpec("old", target_range=(1880, 1900), input_range=(2000, 2030), render_years=(1890,))
    with pytest.raises(ValueError):
        select_probe_rows(ds, spec)                                                # which scene?
    rows = select_probe_rows(ds, RenderProbeSpec.from_dict({**spec.__dict__, "scene_id": 2}))
    assert (ds.views.scene_id[rows] == 2).all()


# --- change events -----------------------------------------------------------------
def test_event_samples_cross_the_change(views):
    from timemachine.data.sampling import event_anchors
    cfg = SamplingConfig(n_inputs=3, n_targets=5, events=((1, 2016.0),), event_prob=1.0)
    assert len(event_anchors(DATES, cfg, 2016.0)) > 0
    ds = DatedViewsDataset(views, cfg, resolution=16)
    assert ds.event_groups == {0: 2016.0}
    for i in range(40):
        rows = ds.indices(i)
        after = ds.dates[rows] >= 2016.0
        assert len(set(after[3:])) == 1 and not (after[:3] == after[3]).any()     # targets one side, inputs other
    plain = DatedViewsDataset(views, CFG, resolution=16)                         # no events: unchanged stream
    assert not plain.event_groups and np.array_equal(plain.indices(0), DatedViewsDataset(views, CFG).indices(0))
