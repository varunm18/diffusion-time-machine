"""Tests for the MegaScenes readers (index helpers, COLMAP binaries, metadata pages)."""
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timemachine.config import scene_prefix
from timemachine.megascenes import colmap
from timemachine.megascenes.index import build_scene_table, file_key, load_recon_metadata
from timemachine.megascenes.metadata import frame_to_records, parse_page, records_to_frame
from timemachine.megascenes.s3 import image_key, raw_metadata_key, reconstruction_key

FIXTURES = Path(__file__).parent / "fixtures"


# --- keys and names ------------------------------------------------------------
def test_scene_prefix_and_keys():
    assert scene_prefix(110925) == "110/925" and scene_prefix(7) == "000/007"
    assert raw_metadata_key(7, "Arco") == "images/000/007/commons/Arco/raw_metadata.json"
    assert image_key(7, "commons/Arco/0/pictures/A B.jpg") == "images/000/007/commons/Arco/0/pictures/A_B.jpg"
    assert reconstruction_key(7, 2, "images.minibin") == "reconstruct_aux/000/007/colmap/2/images.minibin"
    assert reconstruction_key(7, 2, "cameras.bin") == "reconstruct/000/007/colmap/2/cameras.bin"


def test_file_key_unifies_sources():
    decomposed = "Café 1900.jpg"  # 'é' as e + combining accent
    assert file_key("File:Café 1900.jpg") == file_key(f"commons/X/0/pictures/{decomposed}") == "Café_1900.jpg"


# --- recon metadata / scene table -----------------------------------------------
def test_recon_metadata_and_scene_table(tmp_path):
    path = tmp_path / "recon.json"
    path.write_text(json.dumps({"5": ["A", 2, 4, 100, 30, 900], "9": ["B", 1, 12, 50]}))
    recon = load_recon_metadata(path)
    assert recon.to_dict("list")["n_images"] == [4, 30, 12]

    index = pd.DataFrame({
        "cat": ["A", "A", "A", "B", "C"],
        "subcat": ["A", "A_old", "A", "B", "C"],
        "file_key": ["x.jpg", "x.jpg", "y.jpg", "z.jpg", "w.jpg"],  # x.jpg duplicated in A
    })
    table = build_scene_table(index, {"A": 5, "B": 9, "C": 11}, recon).set_index("scene")
    assert table.loc["A", "n_paths"] == 3 and table.loc["A", "n_files"] == 2
    assert table.loc["A", "largest_model"] == 1 and table.loc["A", "largest_model_images"] == 30
    assert table.loc["C", "n_models"] == 0 and pd.isna(table.loc["C", "largest_model"])


# --- COLMAP binaries ------------------------------------------------------------
def _cameras_bin() -> bytes:
    return struct.pack("<Q", 1) + struct.pack("<iiQQ", 3, 2, 800, 600) + struct.pack("<4d", 700, 400, 300, 0.01)


def _images_bytes(with_points: bool) -> bytes:
    out = struct.pack("<Q", 2)
    for image_id, name in ((1, "commons/S/0/pictures/a.jpg"), (2, "commons/S/0/pictures/b_c.jpg")):
        out += struct.pack("<I4d3dI", image_id, 1, 0, 0, 0, 1.0, 2.0, 3.0, 3) + name.encode() + b"\x00"
        if with_points:
            out += struct.pack("<Q", 2) + b"\x00" * 48
    return out


def test_read_cameras():
    (cam,) = colmap.read_cameras_bin(_cameras_bin())
    assert cam.model == "SIMPLE_RADIAL" and (cam.width, cam.height) == (800, 600)
    assert cam.focal_and_center() == (700, 700, 400, 300)


@pytest.mark.parametrize("reader, with_points", [
    (colmap.read_images_bin, True), (colmap.read_images_minibin, False),
])
def test_read_images(reader, with_points):
    images = reader(_images_bytes(with_points))
    assert [im.name.split("/")[-1] for im in images] == ["a.jpg", "b_c.jpg"]
    np.testing.assert_allclose(images[0].camera_center(), [-1.0, -2.0, -3.0])


def test_wrong_reader_is_detected():
    with pytest.raises((ValueError, struct.error)):
        colmap.read_images_minibin(_images_bytes(with_points=True))


# --- Commons metadata pages -----------------------------------------------------
@pytest.fixture(scope="module")
def pages():
    return {x["page"]["title"]: x["page"] for x in json.loads((FIXTURES / "commons_pages.json").read_text())}


def test_parse_page_extracts_fields(pages):
    meta = parse_page(pages["File:A. H. Payne Dresden.jpg"])
    assert meta.date_text == "circa 1851"
    assert meta.date_qs.startswith("P571,+1851")
    assert meta.artist == "Albert Henry Payne" and meta.artist_lifespan == (1812, 1902)
    assert "Frauenkirche Dresden (before 1945)" in meta.categories
    assert meta.file_key == "A._H._Payne_Dresden.jpg"


def test_frame_roundtrip(pages):
    records = [parse_page(p) for p in pages.values()]
    df = records_to_frame(records)
    df["scene_id"], df["subcats"] = 1, [["s"]] * len(df)  # extra columns are ignored
    assert frame_to_records(df) == records


def test_safe_filename():
    from timemachine.config import safe_filename
    assert safe_filename("Kirchplatz:_Brunnen") == "Kirchplatz%3A_Brunnen"
    assert safe_filename('Views_from_"Mirador"') == "Views_from_%22Mirador%22"
    assert safe_filename("Plain_(name),_ok") == "Plain_(name),_ok"
    long = safe_filename("x" * 500)
    assert len(long.encode()) <= 180 and long != safe_filename("x" * 499)
