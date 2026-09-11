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
