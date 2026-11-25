"""Check that a training run would actually start, before it is queued.

A date-conditioned run waits in a queue and then takes hours, so the expensive
failures are the cheap ones: a missing weight file, a dataset that was never built,
an image root that does not exist on this machine. Each of those kills the job
seconds after it starts, and the next attempt goes to the back of the queue.

Every check answers one question - would this stop the run? - and says what to do
about it rather than only that it failed. They are ordered cheapest first and run
independently, so one failure does not hide the others: it is more useful to learn
that three things are missing than to discover them one queue wait at a time.

Two deliberate choices about how the deep checks work:

``check_dataset_loads`` builds a real sample through the repo's own dataset class,
but passes arguments by inspecting its signature rather than hardcoding them. The
dataset gains keyword arguments as the project moves (``with_appearance`` arrived
with appearance conditioning), and a preflight that guesses wrong reports a failure
that is its own fault - the worst possible outcome for a tool whose only job is to
be trusted when it says something is broken.

``check_smoke_train`` shells out to the real training script instead of assembling
a step in process, for the same reason. It reads ``--help`` first and uses only
flags that are actually there, so it tests the command you will submit rather than a
parallel reimplementation that can drift away from it.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# The three snapshots model/loading.py expects to find in a Hugging Face hub cache.
REQUIRED_REPOS = {
    "stabilityai/stable-virtual-camera": "SEVA weights (gated - request access on the model page)",
    "Manojb/stable-diffusion-2-1-base": "the SD 2.1 VAE (a mirror; SEVA's original repo is gone)",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "the CLIP image encoder",
}

# Columns the dataset cannot build a sample without. Deliberately a subset of
# VIEW_COLUMNS: extra columns are harmless, these are not optional.
REQUIRED_VIEW_COLUMNS = (
    "scene_id", "model", "path",
    "camera_width", "camera_height", "fx", "fy", "cx", "cy", "k1", "k2",
    "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "date_lo", "date_hi", "date_mid", "era", "medium",
)

# Below this there is not enough signal for date conditioning to learn anything,
# which is worth knowing before a run rather than after reading its samples.
MIN_OLD_VIEWS = 100
OLD_BEFORE_YEAR = 1945

# An 8-view 576px bf16 step with checkpointing peaked near 14 GB on an A6000.
MIN_GPU_GB = 20


@dataclass
class Check:
    """One answered question about whether a run would start."""

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    skipped: bool = False

    @property
    def mark(self) -> str:
        return "--" if self.skipped else ("ok" if self.ok else "FAIL")


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.skipped]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        width = max((len(c.name) for c in self.checks), default=0)
        lines = [f"  {c.mark:>4}  {c.name:<{width}}  {c.detail}".rstrip() for c in self.checks]
        if self.ok:
            lines += ["", "Ready to submit."]
        else:
            lines += ["", f"{len(self.failures)} blocking problem(s):"]
            lines += [f"  - {c.name}: {c.fix}" for c in self.failures if c.fix]
        return "\n".join(lines)


def _have(module: str) -> bool:
    """Is the module importable, without importing it (torch costs seconds)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_imports() -> list[Check]:
    """The packages a run needs, named one at a time so a missing extra is precise."""
    out = []
    for module, fix, why in (
        ("torch", 'pip install -e ".[torch]"', "the dataset and the model"),
        ("pandas", 'pip install -e ".[torch]"', "reading the view tables"),
        ("seva.model", "bash scripts/setup_seva.sh", "SEVA itself"),
        ("open_clip", 'pip install -e ".[seva]"', "the CLIP conditioner"),
        ("diffusers", 'pip install -e ".[seva]"', "the VAE"),
    ):
        present = _have(module)
        out.append(Check(f"import {module}", present,
                         "" if present else f"needed for {why}", fix))
    return out


def check_gpu() -> Check:
    """A GPU, and whether it is big enough. Absent is fine on a submission node."""
    if not _have("torch"):
        return Check("gpu", True, skipped=True, detail="torch not installed")
    import torch

    if not torch.cuda.is_available():
        return Check("gpu", True, skipped=True,
                     detail="no CUDA here (expected on a login/submission node)")
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return Check("gpu", gb >= MIN_GPU_GB, f"{torch.cuda.get_device_name(0)}, {gb:.0f} GB",
                 f"under ~{MIN_GPU_GB} GB is tight for 8 views at 576px; ask for an A5000 "
                 "(--gres=gpu:rtxa5000:1) or lower --n-targets")


def hub_root(explicit: Path | None = None) -> Path:
    """Where the weights are, preferring the repo's own resolver if it has one."""
    if explicit:
        return Path(explicit)
    try:
        from timemachine.model.loading import default_hub  # type: ignore[attr-defined]

        return Path(default_hub())
    except Exception:                                  # noqa: BLE001 - fall back to the env vars
        for var in ("TIMEMACHINE_HF_HUB", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
            if os.environ.get(var):
                value = Path(os.environ[var])
                return value if value.name == "hub" else value / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"


def check_weights(hub: Path | None = None) -> list[Check]:
    """Each weight repo the loader needs, checked one at a time.

    A repo folder can exist while holding nothing - an interrupted download leaves
    the directory and the refs behind - so this looks for a snapshot with files in
    it rather than for the folder.
    """
    root = hub_root(hub)
    out = [Check("hub cache", root.is_dir(), str(root),
                 f"{root} does not exist; point TIMEMACHINE_HF_HUB at the cache that "
                 "holds the weights, or download them there")]
    if not root.is_dir():
        return out
    for repo, why in REQUIRED_REPOS.items():
        folder = root / f"models--{repo.replace('/', '--')}"
        snapshots = sorted(folder.glob("snapshots/*")) if folder.is_dir() else []
        filled = [s for s in snapshots if next(s.rglob("*"), None) is not None]
        out.append(Check(f"weights {repo.split('/')[-1]}", bool(filled),
                         filled[-1].name[:12] if filled else f"missing - {why}",
                         f"huggingface-cli download {repo} --cache-dir {root}"))
    return out


def _view_table(folder: Path) -> Path | None:
    """The view table in a dataset folder, preferring the cleaned one."""
    return next((folder / name for name in ("views_clean.parquet", "views.parquet")
                 if (folder / name).is_file()), None)


def _resolve(path: str, folder: Path, image_root: Path | None) -> Path | None:
    """Find an image that a table refers to, wherever the table chose to refer to it.

    Tables have carried absolute paths, paths relative to the repo root, and paths
    relative to a shared image cache at different points, and a dataset copied
    between machines keeps whichever it was built with. The run only needs the file
    to be findable, so try the plausible roots instead of insisting on one.

    An absolute path needs its leading directories stripped before a new root can be
    tried, because ``Path("/new/root") / "/old/machine/img.jpg"`` is just the old
    path again - pathlib treats an absolute right-hand side as replacing the left.
    Joining without stripping is why a relocated dataset looks unfixable: the root
    you pass is discarded silently and every image still reports missing.
    """
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate

    # The tail of the path that is worth re-rooting: the bare file name, and the
    # name with its immediate parent, which is what distinguishes images/ from
    # thumbnails/ when a scene folder holds both.
    tails = [Path(candidate.name)]
    if candidate.parent.name:
        tails.append(Path(candidate.parent.name) / candidate.name)
    if not candidate.is_absolute():
        tails.append(candidate)

    bases = ([image_root] if image_root else []) + [folder, folder.parent, Path.cwd()]
    return next((base / tail for base in bases for tail in tails
                 if (base / tail).is_file()), None)


def check_datasets(folders: Sequence[Path], image_root: Path | None = None,
                   sample_images: int = 3, min_old: int = MIN_OLD_VIEWS) -> list[Check]:
    """Each dataset folder: a readable table, the columns, and images that exist."""
    if not folders:
        return [Check("datasets", False, "none given",
                      "pass --dataset data/datasets/<scene>_model0")]
    if not _have("pandas"):
        return [Check("datasets", True, skipped=True, detail="pandas not installed")]
    import pandas as pd

    out: list[Check] = []
    total = old = 0
    for folder in folders:
        name = f"dataset {folder.name}"
        table = _view_table(folder)
        if table is None:
            out.append(Check(name, False, "no views parquet here",
                             f"build it: python scripts/build_scene_dataset.py --scene {folder.name}"))
            continue
        try:
            views = pd.read_parquet(table)
        except Exception as exc:                       # noqa: BLE001 - report, never raise
            out.append(Check(name, False, f"unreadable: {type(exc).__name__}: {exc}",
                             "rebuild the dataset"))
            continue
        missing = [c for c in REQUIRED_VIEW_COLUMNS if c not in views.columns]
        if missing:
            out.append(Check(name, False, f"missing columns: {', '.join(missing[:5])}",
                             "rebuild it with the current scripts/build_scene_dataset.py"))
            continue
        if views.empty:
            out.append(Check(name, False, "table has no rows", "rebuild the dataset"))
            continue
        gone = [p for p in views["path"].head(sample_images)
                if _resolve(str(p), folder, image_root) is None]
        if gone:
            out.append(Check(name, False, f"{len(gone)} of {min(sample_images, len(views))} "
                             "sampled images not found",
                             f"the images this table names are not on this machine "
                             f"(e.g. {gone[0]}); pass --image-root, or re-download them"))
            continue
        total += len(views)
        old += int((views["date_mid"] < OLD_BEFORE_YEAR).sum())
        out.append(Check(name, True, f"{len(views):,} views"))

    if total:
        out.append(Check("old photos", old >= min_old,
                         f"{old:,} of {total:,} views ({old / total:.0%}) before {OLD_BEFORE_YEAR}",
                         f"under ~{min_old} old views there is little for date "
                         "conditioning to learn; add scenes with scripts/plan_dataset.py"))
    return out


def _call_with_supported(factory: Callable, *args, **wanted):
    """Call a factory passing only the keyword arguments it actually declares.

    The alternative - hardcoding today's signature - makes the preflight report its
    own staleness as a project failure.
    """
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):                    # builtins and C callables
        return factory(*args)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(*args, **wanted)
    return factory(*args, **{k: v for k, v in wanted.items() if k in params})


def check_dataset_loads(folders: Sequence[Path], resolution: int = 576,
                        appearance: bool = True) -> Check:
    """Build one real sample. Catches what a column check cannot: a corrupt image,
    an unreadable pose, a resolution the crop cannot satisfy."""
    for module in ("torch", "pandas"):
        if not _have(module):
            return Check("sample builds", True, skipped=True, detail=f"needs {module}")
    if not folders:
        return Check("sample builds", True, skipped=True, detail="no datasets given")
    try:
        from timemachine.data.dataset import DatedViewsDataset
        from timemachine.data.views import load_views

        views = load_views([str(f) for f in folders])
        dataset = _call_with_supported(DatedViewsDataset, views, resolution=resolution,
                                       with_appearance=appearance)
        sample = dataset[0]
        keys = sorted(sample)[:6] if hasattr(sample, "keys") else []
        return Check("sample builds", True,
                     f"{len(dataset)} samples; first has {', '.join(keys)}"
                     + (" ..." if keys else ""))
    except ImportError as exc:
        # The package itself is not importable, which says nothing about the data.
        # Reporting it as a dataset failure would send someone rebuilding datasets
        # that are fine, and the import checks above already name the real cause.
        return Check("sample builds", True, skipped=True,
                     detail=f"cannot import the repo here ({exc.name or exc})")
    except Exception as exc:                           # noqa: BLE001 - the point is to report it
        return Check("sample builds", False, f"{type(exc).__name__}: {exc}",
                     "the dataset cannot produce a sample; this would kill the job at step 0")


def script_flags(script: Path, python: str = "python") -> set[str]:
    """The long flags a script's --help advertises, so the smoke run uses real ones."""
    try:
        helped = subprocess.run([python, str(script), "--help"], capture_output=True,
                                text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return set()
    return set(re.findall(r"--[\w-]+", helped.stdout + helped.stderr))


def check_smoke_train(script: Path, folders: Sequence[Path], extra: Sequence[str] = (),
                      python: str = "python", timeout: int = 1800,
                      step_flag: str = "--max-steps") -> Check:
    """Run the real training script for one step.

    This is the check worth waiting for. It exercises the path a submitted job takes
    - argument parsing, weight loading, the model build, a forward and a backward
    pass, the optimizer - using the script that will actually be submitted.
    """
    if not script.is_file():
        return Check("smoke train", False, f"{script} not found",
                     "point --train-script at the training entrypoint")
    flags = script_flags(script, python)
    if not flags:
        return Check("smoke train", False, f"{script} --help produced nothing",
                     "the script fails before parsing arguments; run it by hand to see why")
    if step_flag not in flags:
        return Check("smoke train", False, f"{script} has no {step_flag}",
                     f"a one-step run needs a step limit; add {step_flag} to the script "
                     "or pass --step-flag with the right name")

    cmd = [python, str(script), step_flag, "1"]
    for flag, values in (("--dataset", [str(f) for f in folders]),):
        if flag in flags:
            cmd += [flag, *values]
    # Skip the things a one-step run has no reason to do, where the script offers it.
    for flag in ("--no-save", "--no-wandb", "--no-eval"):
        if flag in flags:
            cmd.append(flag)
    cmd += [a for a in extra if not a.startswith("--") or a.split("=")[0] in flags]

    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Check("smoke train", False, f"no step finished in {timeout}s",
                     "one step should not take this long; check the GPU is visible "
                     "and the dataset is on fast local storage")
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return Check("smoke train", False, f"exit {done.returncode}: {' / '.join(tail)}",
                     f"this is exactly what the queued job would do; reproduce it with:\n"
                     f"      {shlex.join(cmd)}")
    loss = re.findall(r"loss[\"']?[:= ]+([0-9.]+)", done.stdout, re.I)
    return Check("smoke train", True,
                 f"one step completed" + (f", loss {loss[-1]}" if loss else ""))


def run_checks(folders: Sequence[Path], hub: Path | None = None,
               image_root: Path | None = None, train_script: Path | None = None,
               extra: Sequence[str] = (), appearance: bool = True,
               min_old: int = MIN_OLD_VIEWS,
               python: str = "python", timeout: int = 1800,
               progress: Callable[[str], None] = lambda _: None) -> Report:
    """Every check, cheapest first, each independent of the rest."""
    report = Report()
    report.add(*check_imports())
    report.add(check_gpu())
    report.add(*check_weights(hub))
    report.add(*check_datasets(folders, image_root, min_old=min_old))
    report.add(check_dataset_loads(folders, appearance=appearance))
    if train_script is None:
        report.add(Check("smoke train", True, skipped=True, detail="--no-smoke"))
    else:
        progress(f"running one training step via {train_script} (this takes a few minutes) ...")
        report.add(check_smoke_train(train_script, folders, extra, python, timeout))
    return report
