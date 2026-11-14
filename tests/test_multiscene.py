import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from timemachine.config import DataPaths  # noqa: E402
from timemachine.data.cache import write_cache  # noqa: E402
from timemachine.data.dataset import load_view  # noqa: E402
from timemachine.data.multiscene import MultiSceneDataset, cap_group_share, view_intrinsics  # noqa: E402
from timemachine.data.sampling import SamplingConfig  # noqa: E402

OLD = [1930.0, 1931.0, 1932.0, 1933.0, 1934.0, 1880.0]      # five within +-5 years: old anchors


def _write_model(paths, sid, old_dates, n_modern=12, dup_pairs=()):
    """A model whose latent values encode (scene id, row), so tests can check row alignment."""
    dates = np.concatenate([old_dates, np.full(n_modern, 2015.5)])
    n = len(dates)
    rng = np.random.default_rng(sid)
    quat = rng.normal(size=(n, 4))
    views = pd.DataFrame({
        "scene_id": sid, "model": 0, "file_key": [f"s{sid}_{i}.jpg" for i in range(n)],
        "camera_width": 1600, "camera_height": 1200, "image_width": 1024, "image_height": 768,
        "fx": 1400.0, "fy": 1400.0, "cx": 800.0, "cy": 600.0,
        "qw": quat[:, 0], "qx": quat[:, 1], "qy": quat[:, 2], "qz": quat[:, 3],
        "tx": rng.normal(size=n), "ty": rng.normal(size=n), "tz": rng.normal(size=n),
        "date_lo": dates - 0.5, "date_hi": dates + 0.5, "date_mid": dates,
        "date_confidence": [0.4] + [0.9] * (n - 1), "medium": "photo", "is_monochrome": dates < 1970,
        "dup_cluster": [f"{sid}/c{i}" for i in range(n)],
    })
    for a, b in dup_pairs:
        views.loc[b, "dup_cluster"] = views.loc[a, "dup_cluster"]
    folder = paths.dataset_dir(sid, 0)
    folder.mkdir(parents=True)
    views.to_parquet(folder / "views.parquet", index=False)
    latents = np.zeros((n, 4, 72, 72), dtype=np.float32)
    latents[:, 0, 0, 0], latents[:, 0, 0, 1] = sid, np.arange(n)
    write_cache(folder, views["file_key"].tolist(), latents, np.tile(np.arange(n)[:, None], (1, 1024)))
    return {"scene_id": sid, "scene": f"Scene{sid}", "model": 0, "role": "old_target"}


@pytest.fixture
def paths(tmp_path):
    paths = DataPaths(root=tmp_path)
    paths.tables.mkdir(parents=True)
    rows = [_write_model(paths, 1, OLD), _write_model(paths, 2, OLD, dup_pairs=[(0, 1)]),
            _write_model(paths, 3, np.arange(1860.0, 1970.0, 11.0)), _write_model(paths, 4, OLD)]
    pd.DataFrame(rows).to_parquet(paths.tables / "models_v1.parquet", index=False)
    pd.DataFrame({"scene_id": [1, 2, 3, 4], "cluster": [1, 1, 3, 4],
                  "split": ["train", "train", "train", "test"]}).to_parquet(paths.tables / "splits_v1.parquet")
    return paths


def test_item_matches_the_contract(paths):
    item = MultiSceneDataset(paths, samples_per_epoch=4)[0]
    assert item["latents"].shape == (8, 4, 72, 72) and item["latents"].dtype == torch.float32
    assert item["clip"].shape == (8, 1024) and item["K"].shape == (8, 3, 3) and item["c2w"].shape == (8, 4, 4)
    assert item["is_input"].tolist() == [True] * 3 + [False] * 5
    assert len(item["file_keys"]) == len(set(item["file_keys"])) == 8
    assert torch.isclose(item["c2w"][0, :3, 3].norm(), torch.tensor(2.0), atol=1e-4)   # SEVA normalization


def test_latents_and_keys_come_from_the_same_rows(paths):
    data = MultiSceneDataset(paths, samples_per_epoch=30)
    for index in range(30):
        item = data[index]
        rows = [int(key.split("_")[1].split(".")[0]) for key in item["file_keys"]]
        assert item["latents"][:, 0, 0, 1].tolist() == rows and item["clip"][:, 0].tolist() == rows
        assert set(item["latents"][:, 0, 0, 0].tolist()) == {float(item["scene_id"])}   # one model per sample


def test_targets_share_the_drawn_era_and_years_stay_in_their_interval(paths):
    data = MultiSceneDataset(paths, samples_per_epoch=60)
    eras = set()
    for index in range(60):
        item = data[index]
        eras.add(item["target_era"])
        years = item["year"][~item["is_input"]]
        if item["target_era"] == "2000+":
            assert ((years >= 2015.0) & (years <= 2016.0)).all()
        else:
            assert ((years >= 1929.5) & (years <= 1934.5)).all()
    assert eras == {"1900-1944", "2000+"}                      # both eras are drawn


def test_low_confidence_dates_are_flagged_unknown(paths):
    data = MultiSceneDataset(paths, samples_per_epoch=80)
    for index in range(80):
        item = data[index]
        for key, valid in zip(item["file_keys"], item["year_valid"].tolist()):
            assert valid == (not key.endswith("_0.jpg"))       # row 0 of every model has confidence 0.4


def test_near_duplicates_never_share_a_sample(paths):
    data = MultiSceneDataset(paths, samples_per_epoch=200)
    for index in range(200):
        keys = data[index]["file_keys"]
        assert not {"s2_0.jpg", "s2_1.jpg"} <= set(keys)


def test_split_filter_and_determinism(paths):
    train, test = MultiSceneDataset(paths, samples_per_epoch=40), MultiSceneDataset(paths, "test", samples_per_epoch=40)
    assert {train[i]["scene_id"] for i in range(40)} <= {1, 2, 3}
    assert {test[i]["scene_id"] for i in range(40)} == {4}
    assert train[7]["file_keys"] == MultiSceneDataset(paths, samples_per_epoch=40)[7]["file_keys"]
    train.set_epoch(1)
    assert any(train[i]["file_keys"] != MultiSceneDataset(paths, samples_per_epoch=40)[i]["file_keys"] for i in range(5))


def test_stale_cache_is_rejected(paths):
    folder = paths.dataset_dir(1, 0)
    views = pd.read_parquet(folder / "views.parquet")
    views.iloc[::-1].to_parquet(folder / "views.parquet", index=False)      # rows reordered after caching
    with pytest.raises(ValueError, match="cache"):
        MultiSceneDataset(paths)


def test_works_in_dataloader_workers(paths):
    data = MultiSceneDataset(paths, samples_per_epoch=8)
    data[0]                                                     # opens memory maps in this process
    assert pickle.loads(pickle.dumps(data))._arrays == {}
    loader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert batch["latents"].shape == (4, 8, 4, 72, 72) and len(batch["file_keys"]) == 8


def test_view_intrinsics_equals_load_view(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (1024, 683)).save(path)
    row = pd.Series({"image_path": str(path), "image_width": 1024, "image_height": 683, "camera_width": 3000,
                     "camera_height": 2000, "fx": 2500.0, "fy": 2510.0, "cx": 1490.0, "cy": 1004.0})
    K_pixels = load_view(row, 576)[1]
    K_pixels[0, :] /= 576
    K_pixels[1, :] /= 576
    assert np.allclose(view_intrinsics(row), K_pixels)


def test_cap_group_share():
    p = cap_group_share(np.array([8.0, 8.0, 1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 2, 3, 4]), 0.25)
    assert abs(p.sum() - 1) < 1e-9 and abs(p[:2].sum() - 0.25) < 1e-9 and np.allclose(p[2:], 0.1875)
    even = cap_group_share(np.ones(4), np.arange(4), 0.25)
    assert np.allclose(even, 0.25)
    assert np.allclose(cap_group_share(np.array([3.0, 1.0]), np.array([0, 1]), 0.25), [0.75, 0.25])   # unsatisfiable
