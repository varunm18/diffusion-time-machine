#!/usr/bin/env python
"""Mirror a multi-scene run (``scripts/mvgen/train_multi.py``) to Weights & Biases, from its files.

The trainer itself does not talk to wandb; this script reads what the run writes and logs it, so it can
start at any time, also after the run, and the training jobs need no network access or change:

* ``<run>/log.jsonl``: every training line as ``train/*`` (loss, gradient norm, seconds per step, the
  date's share of the timestep embedding, and a smoothed loss) and every probe line as
  ``probe/<probe>/*`` (Notre-Dame's held-out and training view sets: loss at the true date, relative loss
  change with a wrong and with no date, embedding norms);
* ``--evals <folder>/<prefix>step<N>/results.json`` (``scripts/mvgen/eval_dates.py``, experiment look1's
  read-outs on Notre-Dame) as ``eval/*`` at step N: fire-state flip, wrong-date cost inside minus outside
  the region, the date's effect on the look, look steering, PSNR;
* ``--sweeps <folder>/step<N>_*/<scene>/set0.jpg`` (date sweeps of ``scripts/mvgen/render.py``) as images
  ``sweeps/<scene>`` at step N.

Each kind has its own step axis (``train/step``, ``eval/step``), so evaluations that finish after later
training steps are logged at their own step. What was sent is recorded in ``<run>/wandb_sync.json``
(the wandb run id, lines and files done), so the script can be stopped and started again; ``--follow``
keeps polling for new lines and files. Credentials: the usual wandb login (``~/.netrc``).

    python scripts/mvgen/wandb_sync.py --run data/experiments/mvgen/multi/ms1 --project diffusion-time-machine \\
        --evals data/experiments/mvgen/eval_look1 --eval-prefix ms1_ --sweeps data/experiments/mvgen/multi/ms1/sweeps --follow
"""
from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import wandb

TRAIN_KEYS = ("loss", "grad_norm", "s_per_step", "date_temb_ratio")
PROBE_KEYS = ("loss_true", "rel_gap_swapped", "rel_gap_unknown", "date_rms", "date_temb_ratio", "look_rms",
              "look_temb_ratio", "sum_temb_ratio")
# look1 read-outs (results.json "look1"): wandb name -> path to a {"mean", "se"} entry
EVAL_KEYS = {
    "fire_state_flip": ("R2", "flip", "all"),
    "fire_state_flip_modern_targets": ("R2", "flip", "modern"),
    "fire_state_flip_after_fire_targets": ("R2", "flip", "after"),
    "wrong_date_cost_inside_minus_outside": ("R2", "loss_inside_minus_outside", "all"),
    "date_look_shift_sd": ("R3", "cross", "all"),
    "within_period_look_shift_sd": ("R3", "within", "all"),
    "psnr_references_look": ("R6", "base_whole", "all"),
    "psnr_own_look": ("R6", "own_whole", "all"),
    "grey_code_chroma": ("R1", "grey_chroma", "all"),
    "old_date_colour_code_chroma": ("R5", "old_date_colour_code"),
}


def step_of(path: Path) -> int:
    """The step number in a path's name (``...step012000...``); -1 without one."""
    match = re.search(r"step(\d+)", str(path.name) if path.name != "results.json" else path.parent.name)
    return int(match.group(1)) if match else -1


def entry(tree: dict, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    if isinstance(tree, dict):
        tree = tree.get("mean", tree.get("chroma"))
    return float(tree) if isinstance(tree, (int, float)) else None


class Sync:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.state_path = args.run / "wandb_sync.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("id", uuid.uuid4().hex[:8])
        self.state.setdefault("lines", 0)
        self.state.setdefault("files", [])
        self.state.setdefault("last_step", -1)
        self.smooth = self.state.get("smooth")
        self.wb = wandb.init(project=args.project, entity=args.entity, name=args.name or args.run.name,
                             id=self.state["id"], resume="allow", dir=str(args.run), config=self.config())
        for axis, pattern in (("train/step", "train/*"), ("train/step", "probe/*"), ("eval/step", "eval/*"),
                              ("eval/step", "sweeps/*")):
            wandb.define_metric(axis, hidden=True)
            wandb.define_metric(pattern, step_metric=axis)
        self.save()

    def config(self) -> dict:
        """The run's settings (its ``config.json``), as the wandb run's config."""
        path = self.args.run / "config.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        self.state["smooth"] = self.smooth
        self.state_path.write_text(json.dumps(self.state))

    def lines(self) -> int:
        path = self.args.run / "log.jsonl"
        if not path.exists():
            return 0
        rows = path.read_text().splitlines()
        new = rows[self.state["lines"]:]
        if new and not rows[-1].endswith("}"):
            new = new[:-1]                     # a line still being written
        sent = 0
        for text in new:
            self.state["lines"] += 1
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            step = int(row.get("step", -1))
            if "probe" in row:
                data = {f"probe/{row['probe']}/{k}": row[k] for k in PROBE_KEYS if k in row}
            else:
                if step <= self.state["last_step"]:
                    continue                   # steps trained again after a resume: keep the first pass
                self.state["last_step"] = step
                data = {f"train/{k}": row[k] for k in TRAIN_KEYS if k in row}
                if "loss" in row:
                    self.smooth = row["loss"] if self.smooth is None else 0.98 * self.smooth + 0.02 * row["loss"]
                    data["train/loss_smooth"] = self.smooth
            if data:
                wandb.log({"train/step": step, **data})
                sent += 1
        return sent

    def evals(self) -> int:
        sent = 0
        if self.args.evals is None:
            return 0
        for path in sorted(self.args.evals.glob(f"{self.args.eval_prefix}step*/results.json"), key=step_of):
            key = str(path)
            match = re.search(r"step(\d+)", path.parent.name)
            if key in self.state["files"] or not match:
                continue
            if time.time() - path.stat().st_mtime < 120:
                continue                       # still being written
            look1 = json.loads(path.read_text()).get("look1", {})
            data = {f"eval/{name}": value for name, where in EVAL_KEYS.items()
                    if (value := entry(look1, where)) is not None}
            if data and all(k in look1 for k in ("R2", "R6")):
                wandb.log({"eval/step": int(match.group(1)), **data})
                self.state["files"].append(key)
                sent += 1
        return sent

    def sweeps(self) -> int:
        sent = 0
        if self.args.sweeps is None:
            return 0
        for folder in sorted(self.args.sweeps.glob("step*"), key=step_of):
            match = re.match(r"step(\d+)", folder.name)
            if not match:
                continue
            for sheet in sorted(folder.glob("*/set0.jpg")):
                key = str(sheet)
                done = sheet.parent / "results.json"
                if key in self.state["files"] or not done.exists():
                    continue
                label = folder.name.split("_", 1)[1] if "_" in folder.name else "references"
                wandb.log({"eval/step": int(match.group(1)),
                           f"sweeps/{sheet.parent.name}": wandb.Image(str(sheet), caption=f"{folder.name}, look {label}")})
                self.state["files"].append(key)
                sent += 1
        return sent

    def once(self) -> int:
        sent = self.lines() + self.evals() + self.sweeps()
        self.save()
        return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder of train_multi.py")
    parser.add_argument("--project", default="diffusion-time-machine")
    parser.add_argument("--entity", help="wandb entity (default: the logged-in user's)")
    parser.add_argument("--name", help="wandb run name (default: the run folder's name)")
    parser.add_argument("--evals", type=Path, help="folder of eval_dates.py outputs")
    parser.add_argument("--eval-prefix", default="", help="only outputs whose folder starts with this")
    parser.add_argument("--sweeps", type=Path, help="folder of step*/<scene>/ date sweeps")
    parser.add_argument("--follow", action="store_true", help="keep polling for new lines and files")
    parser.add_argument("--poll", type=float, default=120, help="seconds between polls with --follow")
    args = parser.parse_args()
    args.run = args.run.resolve()
    sync = Sync(args)
    print(f"[wandb] {sync.wb.url}", flush=True)
    while True:
        sent = sync.once()
        print(f"[wandb] {time.strftime('%H:%M:%S')} logged {sent} records (log lines read {sync.state['lines']})",
              flush=True)
        if not args.follow:
            break
        time.sleep(args.poll)
    wandb.finish()


if __name__ == "__main__":
    main()
