# Diffusion Time Machine

Generate historical or imagined views of a scene with a pretrained Stable Diffusion model.

## Requirements

- Python 3.11 or 3.12 (recommended; required by Metal packages like `msplat`)
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
python3.12 -m venv .venv
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

## 3D Scene Reconstruction & Gaussian Splatting

Reconstruct 3D scenes from images, train 3D Gaussian Splats, and visualize them in Rerun:

```bash
# Run the full end-to-end 3D pipeline
python scripts/run_3d_pipeline.py \
  --image_dir data/sample_images/riverview \
  --workspace data/workspaces/riverview \
  --view_reconstruction \
  --msplat \
  --visualize
```

You can also import components directly into Python:

```python
from src.viz import DUST3R, MSplat, visualize

# 1. Pose estimation & point cloud
dust3r = DUST3R()
dust3r.run(image_dir="data/sample_images/riverview", output_workspace="workspace/riverview")

# 2. Metal 3DGS optimization
splatter = MSplat()
splatter.train("workspace/riverview")

# 3. Interactive visualization in Rerun
visualize("workspace/riverview")
```

## Deactivate the environment

When you are finished:

```bash
deactivate
```
