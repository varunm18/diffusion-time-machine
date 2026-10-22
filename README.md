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

Reconstruct 3D scenes from images and visualize them as 3D Gaussian Splats:

```bash
python 3d_visualizer/main.py \
  --image_dir 3d_visualizer/testing/images/riverview \
  --workspace 3d_visualizer/testing/test_ws \
  --view_output \
  --msplat \
  --visualize
```

For full options and step-by-step commands, see [3d_visualizer/README.md](3d_visualizer/README.md).

## Deactivate the environment

When you are finished:

```bash
deactivate
```
