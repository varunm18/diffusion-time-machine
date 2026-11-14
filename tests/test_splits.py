import pandas as pd

from timemachine.data.splits import choose_holdout, find_leaks, scene_clusters


def test_scenes_sharing_files_form_one_cluster():
    files = {10: {"a", "b"}, 20: {"b", "c"}, 30: {"c", "d"}, 40: {"x"}, 50: {"y", "z"}}
    assert scene_clusters(files) == {10: 10, 20: 10, 30: 10, 40: 40, 50: 50}   # 10-20-30 by a chain


def test_min_shared_ignores_a_single_stray_file():
    files = {1: {"a", "b", "c"}, 2: {"a", "q"}, 3: {"b", "c", "r"}}
    assert scene_clusters(files, min_shared=2) == {1: 1, 2: 2, 3: 1}


def test_holdout_is_stable_spread_out_and_spares_big_clusters():
    clusters = pd.DataFrame({"cluster": range(100, 131), "old_views": [900] + list(range(30, 0, -1))})
    held = choose_holdout(clusters, fraction=0.1)
    assert held == choose_holdout(clusters.sample(frac=1.0, random_state=3), fraction=0.1)
    assert len(held) == 3 and 100 not in held                 # 30 eligible -> 3 groups of 10
    by_rank = sorted(clusters.set_index("cluster").loc[held, "old_views"], reverse=True)
    assert by_rank[0] > 20 >= by_rank[1] > 10 >= by_rank[2]   # one from each third


def test_find_leaks():
    views = pd.DataFrame({"file_key": ["a", "a", "b", "c", "c"], "split": ["train", "train", "test", "train", "test"]})
    assert find_leaks(views) == ["c"]
