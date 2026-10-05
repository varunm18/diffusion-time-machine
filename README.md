# Diffusion Time Machine

Generate historical or imagined views of a scene with a pretrained Stable Diffusion model.

## Requirements

- Python 3.10 or newer
- ~ 5 GB of disk space for the model download

## Setup

Clone the repository and open a terminal in its directory:

```bash
git clone https://github.com/varunm18/diffusion-time-machine.git
cd diffusion-time-machine
```

Create and activate a virtual environment.

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate an image

Run the script with the default historical-scene prompt:

```bash
python generate.py
```

Or provide your own prompt and output filename:

```bash
python generate.py \
	"A realistic historical photograph of the Eiffel Tower in winter, 1889" \
	--output eiffel-tower.png
```

The first run downloads `runwayml/stable-diffusion-v1-5` from Hugging Face and may take several minutes. Later runs reuse the cached model.

## Deactivate the environment

When you are finished:

```bash
deactivate
```

## Data pipeline: dated MegaScenes

The `timemachine` package (in `src/`) builds a table of **when each MegaScenes
image was made** (as a date interval with a confidence), **what kind of image it
is** (photo, postcard, painting, ...), and **whether it has a camera pose** in
MegaScenes' COLMAP reconstructions. Everything is read from the public
MegaScenes bucket; no images are downloaded for this step.

### Setup (uv)

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"          # package + pytest
.venv/bin/python -m pytest          # run the tests
```

Data goes to `data/` in the repo (git-ignored). Set `TIMEMACHINE_DATA=/some/path`
to put it elsewhere.

### Run

```bash
python scripts/fetch_index.py                                        # 0. index files + scene table (~310 MB, once)
python scripts/process_scenes.py --scene-file configs/scenes_pilot.txt  # 1-3. metadata, poses, dates per scene
python scripts/census.py                                             # 4. era counts + go/no-go summary
uv pip install -e ".[torch]"                                         # torch + pillow for the dataset
python scripts/build_scene_dataset.py --scene Cathédrale_Notre-Dame_de_Paris --max-modern 800
                                                                     # 5. single-scene dataset + previews
```

For a full-scale census, submit `slurm/census_cpu.sbatch` (CPU job) instead of
running on a submission node.

`process_scenes.py` also accepts `--scenes NAME_OR_ID ...` or `--top N` (the N
scenes with the largest reconstructions). Every step is cached per scene, so
reruns only do new work.

### Outputs (`data/tables/`)

| File | One row per | Contents |
|---|---|---|
| `scenes.parquet` | scene | image counts (raw and deduplicated), COLMAP model sizes |
| `image_metadata/<scene_id>.parquet` | unique Commons file | date fields, categories, EXIF, author, license |
| `poses/<scene_id>.parquet` | registered image entry | COLMAP model, intrinsics, world-to-camera pose |
| `dates/<scene_id>.parquet` | unique Commons file | date interval, confidence, source, flags, medium, color hint |
| `census_scenes.csv`, `census_models.csv` | scene / model | files per era, posed files per era, gate results |
| `../datasets/<id>_model<m>/views.parquet` | view | posed + dated photo of one model, with local image path |

### Code layout

```
src/timemachine/
  config.py          paths, URLs, User-Agent
  megascenes/        s3 (bucket access), index, metadata (Commons records), colmap, poses
  dating/            interval types, sources/ (one parser per evidence source),
                     resolve (combine evidence), medium (image type), pipeline
  analysis/census.py era counts and the go/no-go gate
  data/              view table, image cache, SEVA camera conventions, sampling, torch Dataset
scripts/             thin command-line entry points, one per step
tests/               pytest; fixtures are real Commons pages
docs/                proposal, literature review, findings log, plans
```

See `docs/findings.md` for what we have learned about the data so far.
