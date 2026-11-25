# Preflight: check a run before you queue it

A training run waits in a queue and then takes hours. The failures that cost the
most are the cheapest ones - a weight file that never finished downloading, a
dataset that was never built, images that live on a different machine than the
table says. Each kills the job seconds after it starts, and the retry goes to the
back of the queue.

```bash
# On the login node, before you sbatch anything:
python scripts/preflight.py --dataset data/datasets/*_model0
```

```
Preflight: 28 dataset folder(s)

    ok  import torch
    ok  import pandas
  FAIL  import seva.model                      needed for SEVA itself
    ok  import open_clip
    ok  import diffusers
    --  gpu                                    no CUDA here (expected on a login/submission node)
    ok  hub cache                              /fs/class-projects/fall2026/aashok/hf/hub
  FAIL  weights stable-virtual-camera          missing - SEVA weights (gated ...)
    ok  weights stable-diffusion-2-1-base      a1b2c3d4e5f6
    ok  weights CLIP-ViT-H-14-laion2B-s32B-b79K  9f8e7d6c5b4a
    ok  dataset Amsterdam_..._model0           412 views
    ...
    ok  old photos                             1,284 of 10,182 views (13%) before 1945

2 blocking problem(s):
  - import seva.model: bash scripts/setup_seva.sh
  - weights stable-virtual-camera: huggingface-cli download stabilityai/stable-virtual-camera --cache-dir ...
```

It exits non-zero when anything would stop the run, so a job script can gate on it.
`slurm/train_dated_seva_class.sbatch` does exactly that.

## What it checks

| Check | Catches |
| --- | --- |
| `import torch`, `seva.model`, `open_clip`, `diffusers` | a missing extra, named one package at a time |
| `gpu` | no GPU, or one too small for 8 views at 576px (skipped on a login node) |
| `hub cache`, `weights …` | each weight repo separately, including a download that left an empty snapshot behind |
| `dataset …` | no view table, an unreadable one, missing columns, no rows, images not on this machine |
| `old photos` | too few pre-1945 views for date conditioning to learn anything |
| `sample builds` | a corrupt image or an unusable pose - things a column check cannot see |
| `smoke train` | argument parsing, weight loading, the model build, one forward and backward pass |

Every check runs independently, so one failure does not hide the others. Finding
out that three things are broken is worth more than finding them one queue wait at
a time.

## The slow check

`--smoke` runs the real training script for one step. It is off by default: the
value of running on a login node is catching the cheap failures before you reach a
GPU, and a one-step rehearsal needs the GPU anyway.

```bash
srun --partition=class --account=class --qos=default \
     --gres=gpu:rtxa5000:1 --mem=32G --time=01:00:00 --pty bash
python scripts/preflight.py --dataset "$DATA_ROOT"/datasets/*_model0 \
    --smoke --train-arg --appearance-remove-era
```

It shells out to `scripts/train_dated_seva.py` rather than assembling a step in
process, and reads `--help` first so it only passes flags that are really there. So
it tests the command you are about to submit, not a parallel reimplementation that
can drift from it. Flags you pass with `--train-arg` that the script does not
advertise are dropped rather than failing the check, which keeps the preflight
usable across branches where the trainer has different options.

When the step fails, the fix line gives the exact command back, so you can
reproduce it by hand without reconstructing the arguments.

## Relocated datasets

The tables record where the images were when the dataset was built. Copy a dataset
between machines and every image reports missing. `--image-root` re-roots them:

```bash
python scripts/preflight.py --dataset "$DATA_ROOT"/datasets/*_model0 \
    --image-root "$DATA_ROOT/images"
```

A detail worth knowing, because it is a trap: `Path("/new/root") / "/old/path/img.jpg"`
returns `/old/path/img.jpg`. Pathlib treats an absolute right-hand side as replacing
the left, so joining a root onto an absolute path does nothing at all - the root is
discarded and every image still reports missing. The resolver strips the leading
directories first and tries the file name both bare and with its immediate parent,
which is what keeps `images/img0.jpg` from matching `thumbnails/img0.jpg`.

## Why it does not hardcode the dataset signature

`sample builds` constructs a real sample through `DatedViewsDataset`, but passes
arguments by inspecting its signature rather than assuming them. The dataset gains
keyword arguments as the project moves - `with_appearance` arrived with appearance
conditioning, and only exists on some branches. A preflight that guessed wrong
would report a failure that was its own fault, which is the worst outcome for a
tool whose only job is to be believed when it says something is broken.

The same reasoning is why `REQUIRED_VIEW_COLUMNS` is a deliberate subset of
`VIEW_COLUMNS`: extra columns are harmless, so only the ones a sample cannot be
built without are treated as required.
