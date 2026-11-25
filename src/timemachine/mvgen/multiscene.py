"""Training samples drawn from many scenes at once (the first multi-scene run, December 2026).

Each scene stays an :class:`timemachine.mvgen.data.MVGenScene` folder with its own view table, depth,
look codes, scene scale (``scene.json``) and period edges (``scene.json``'s ``period_edges``; 1970 when
absent). A sample never mixes scenes. :class:`SceneMixture` adds what a mixture needs:

* **Which scene.** Sample ``number`` of a run first picks a scene, with probability proportional to the
  square root of its number of training views (so large scenes do not drown small ones), except that the
  scenes named in ``fixed_shares`` get exactly that share (Notre-Dame, so it can be compared with the
  single-scene runs at a matched number of samples). The pick, and the sample's index inside the scene,
  come from a generator seeded by ``(seed, number)`` alone: a resumed run and every data-loading process
  draw the same samples, and data-parallel ranks take disjoint numbers (rank, rank + world, ...).
* **Hold-out.** A scene with changed-region weights (Notre-Dame) holds out whole day-and-photographer
  components as in experiment look1 (``holdout_by_component``; with look1's settings the same views,
  so look1's evaluation sets stay held out); every other scene holds out 10% of its views per era at
  random (``HoldoutConfig(by="view")``), because the component rule needs side views. Whole held-out
  scenes are simply not in the mixture.
* **Look codes.** One standardisation for all scenes (:class:`~timemachine.mvgen.look.LookStats` of the
  training views of every scene together), so a code means the same colours in every scene. Arm D's
  sepia conversion needs tone curves of real toned prints; a scene without toned prints of its own
  borrows the pooled curves of all scenes.

The training step itself is unchanged (``timemachine.mvgen.step.training_loss``), called with the
sample's own period edges.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from timemachine.data.holdout import HoldoutConfig
from timemachine.mvgen.data import MVGenScene, SCENE_FILE, ViewSamplingConfig
from timemachine.mvgen.look import LookStats, TrainingLooks, read_class_maps

DEFAULT_EDGES = (1970.0,)
KNOWN_EDGES = {25512: (1970.0, 2019.2877)}     # scenes built before scene.json named its edges: Notre-Dame (the fire)
LOOK1_HOLDOUT = dict(fraction=0.1, by="component", seed=7, max_unit=12, side_share=0.05, unit_periods=(0,))


def scene_edges(folder: Path) -> tuple[float, ...]:
    """The period edges a scene folder's ``scene.json`` names (1970 when it names none)."""
    scene = json.loads((Path(folder) / SCENE_FILE).read_text(encoding="utf-8"))
    edges = scene.get("period_edges", KNOWN_EDGES.get(int(scene["scene_id"]), DEFAULT_EDGES))
    return tuple(float(edge) for edge in edges)


@dataclass
class LookSettings:
    """``LossConfig``'s look settings, as TrainingLooks takes them."""

    dropout: float = 0.5
    unspecified: float = 0.3
    grey: float = 0.1
    toned: float = 0.05


@dataclass
class SceneMixture:
    """Many scene folders as one stream of numbered training samples (see the module docstring)."""

    folders: Sequence[Path]
    sampling: ViewSamplingConfig = field(default_factory=ViewSamplingConfig)
    seed: int = 0
    resolution: int = 512
    camera_longest_side: float = 5.0
    fixed_shares: Mapping[str, float] = field(default_factory=dict)      # folder name -> share of samples
    look: LookSettings | None = None
    holdout_fraction: float = 0.1

    def __post_init__(self):
        self.scenes: list[MVGenScene] = []
        self.names: list[str] = []
        for folder in self.folders:
            folder = Path(folder)
            config = replace(self.sampling, period_edges=scene_edges(folder))
            scene = MVGenScene(folder, config, resolution=self.resolution, seed=self.seed,
                               camera_longest_side=self.camera_longest_side, require_depth=True, scale="scene")
            if scene.has_regions:
                holdout = HoldoutConfig(**{**LOOK1_HOLDOUT, "era_edges": config.period_edges})
            else:
                holdout = HoldoutConfig(fraction=self.holdout_fraction, seed=7)
            scene.hold_out(scene.holdout_rows(holdout))
            if len(scene.anchors) == 0:
                print(f"[mixture] {folder.name}: no sample can be drawn after the hold-out; left out", flush=True)
                continue
            self.scenes.append(scene)
            self.names.append(folder.name)
        if not self.scenes:
            raise ValueError("no scene of the mixture can give a sample")
        self.shares = self._shares()
        self.looks = self._looks() if self.look is not None else None

    # ------------------------------------------------------------------ set-up
    def _shares(self) -> np.ndarray:
        size = np.array([np.sqrt(scene.target_ok.sum()) for scene in self.scenes], dtype=np.float64)
        fixed = np.array([self.fixed_shares.get(name, np.nan) for name in self.names], dtype=np.float64)
        rest = 1.0 - np.nansum(fixed)
        if rest <= 0:
            raise ValueError("the fixed shares leave nothing for the other scenes")
        free = np.isnan(fixed)
        shares = np.where(free, 0.0, fixed)
        shares[free] = rest * size[free] / size[free].sum()
        return shares / shares.sum()

    def _looks(self) -> list[TrainingLooks]:
        """One TrainingLooks per scene, all standardised with the statistics of every scene's training views."""
        missing = [name for name, scene in zip(self.names, self.scenes) if not scene.has_look]
        if missing:
            raise ValueError(f"no look codes in {missing}: run scripts/mvgen/prepare_look.py on them")
        codes = []
        for scene in self.scenes:
            look = scene.look
            codes.append(np.asarray(look["codes"])[scene.target_ok & np.asarray(look["computed"], dtype=bool)])
        self.look_stats = LookStats.from_codes(np.concatenate(codes))
        settings = self.look
        converts = settings.grey + settings.toned > 0
        pooled = None
        if converts:        # the tone curves of every scene's toned training prints, for scenes that have none
            pooled = np.concatenate([
                np.asarray(scene.look["toned_curves"])[
                    scene.target_ok[np.asarray(scene.look["toned_rows"], dtype=np.int64)]]
                for scene in self.scenes if len(scene.look["toned_rows"])])
        looks = []
        for scene in self.scenes:
            class_maps = read_class_maps(scene.dataset_dir, scene.views["file_key"].to_numpy()) if converts else None
            own = (len(scene.look["toned_rows"]) > 0
                   and scene.target_ok[np.asarray(scene.look["toned_rows"], dtype=np.int64)].any())
            looks.append(TrainingLooks(scene.look, scene.target_ok, self.look_stats, seed=self.seed,
                                       dropout=settings.dropout, unspecified=settings.unspecified,
                                       grey=settings.grey, toned=settings.toned if own else 0.0,
                                       class_maps=class_maps))
            if converts and not own and settings.toned > 0:      # borrow the pooled curves
                looks[-1].toned, looks[-1].curves = settings.toned, pooled
        return looks

    # ------------------------------------------------------------------ samples
    def pick(self, number: int) -> tuple[int, int]:
        """The scene and the scene-sampler index of sample ``number``."""
        rng = np.random.default_rng((self.seed, 7919, int(number)))
        scene = int(rng.choice(len(self.scenes), p=self.shares))
        return scene, int(rng.integers(len(self.scenes[scene])))

    def __getitem__(self, number: int) -> dict:
        s, index = self.pick(number)
        scene = self.scenes[s]
        sample = scene.make_sample(*scene.draw(index, epoch=int(number)))
        if self.looks is not None:
            sample = self.looks[s](sample, int(number), index)
        sample["scene"] = s
        sample["period_edges"] = scene.config.period_edges
        return sample

    def summary(self) -> str:
        lines = [f"{'scene':40s} {'training':>8s} {'held out':>8s} {'anchors':>7s} {'share':>6s}  edges"]
        for name, scene, share in zip(self.names, self.scenes, self.shares):
            lines.append(f"{name:40s} {int(scene.target_ok.sum()):8d} {int(scene.heldout.sum()):8d} "
                         f"{len(scene.anchors):7d} {share:6.3f}  {list(scene.config.period_edges)}")
        return "\n".join(lines)
