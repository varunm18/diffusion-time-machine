"""Contact sheets of dataset samples, for eyeballing what the model will see."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sample_contact_sheet(sample: dict, thumb: int = 256) -> Image.Image:
    """One row per sample view: inputs (green label) then targets (red label), with dates."""
    images = sample["images"]                       # (V, 3, S, S) in [-1, 1]
    n = images.shape[0]
    sheet = Image.new("RGB", (thumb * n, thumb + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(n):
        pixels = ((images[i].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(pixels).resize((thumb, thumb)), (i * thumb, 22))
        lo, hi = sample["date"][i].tolist()
        label = f"{'IN' if sample['is_input'][i] else 'TGT'} {int(lo)}" + (f"-{int(hi - 1e-6)}" if hi - lo > 1.01 else "")
        draw.text((i * thumb + 4, 4), label, fill=(0, 140, 0) if sample["is_input"][i] else (200, 0, 0))
    return sheet


def save_sample_previews(dataset, out_dir: Path, n: int = 4) -> list[Path]:
    """Save contact sheets of the first ``n`` samples of a dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = out_dir / f"sample_{i}.jpg"
        sample_contact_sheet(dataset[i]).save(path, quality=90)
        paths.append(path)
    return paths
