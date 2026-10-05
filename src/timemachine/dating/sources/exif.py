"""Capture dates from EXIF, with detection of scans and reproductions.

EXIF ``DateTimeOriginal`` is very reliable for born-digital photos, but for a
scanned or re-photographed old image it is the *digitization* date (we have
seen a 1900 photo with EXIF from a 1999 Olympus, museum scans from 2011
Hasselblad and Phase One backs, a 2012 HP scanner). This parser reports the
date and flags devices that suggest a scan/reproduction; the resolver decides.
"""
from __future__ import annotations

import re

from timemachine.dating.interval import Confidence, DateCandidate, DateInterval, Precision, Source

# Scanners, scanning software and camera backs typically used for reproductions.
_REPRO_DEVICE = re.compile(
    r"scan|epson|perfection|canoscan|hp pst|scanjet|plustek|fujitsu fi-|kodak i\d|"
    r"flextight|imacon|hasselblad|phase ?one|\biq[1-4]\b|\bp ?[2-6]\d\+|leaf aptus|sinar|"
    r"cruse|zeutschel|book2net|metis|eskoscan|vuescan|silverfast|digitali[sz]",
    re.I,
)
_EXIF_DATE = re.compile(r"^(\d{4})[:\-](\d{2})[:\-](\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2}))?")
FIRST_DIGITAL_YEAR = 1990  # EXIF dates before this were typed in by hand (or are wrong)


def device_string(exif: dict[str, str]) -> str:
    """'Make Model Software' as one string (empty if unknown)."""
    return " ".join(exif.get(k, "") for k in ("Make", "Model", "Software")).strip()


def is_reproduction_device(exif: dict[str, str]) -> bool:
    """True if the EXIF device looks like a scanner or a reproduction camera."""
    return bool(_REPRO_DEVICE.search(device_string(exif)))


def parse_exif(exif: dict[str, str]) -> DateCandidate | None:
    """Candidate from EXIF DateTimeOriginal (or DateTimeDigitized as a fallback).

    Flags: ``digitized`` (only DateTimeDigitized present), ``scanner`` (device
    looks like a scan/reproduction setup), ``placeholder`` (1 January,
    the default of unset camera clocks), ``pre_digital`` (year before 1990: typed in by
    hand, plausible but unverifiable).
    """
    flags: list[str] = []
    raw = exif.get("DateTimeOriginal")
    if not raw:
        raw = exif.get("DateTimeDigitized")
        flags.append("digitized")
    m = _EXIF_DATE.match((raw or "").strip())
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        interval = DateInterval.from_date(year, month, day)
    except ValueError:  # 0000:00:00 and friends
        return None

    if is_reproduction_device(exif):
        flags.append("scanner")
    if (month, day) == (1, 1):  # unset camera clocks default to 1 January (2000, 2001, ...)
        flags.append("placeholder")
    if year < FIRST_DIGITAL_YEAR:
        flags.append("pre_digital")

    has_camera = bool(exif.get("Make") or exif.get("Model"))
    if {"placeholder", "pre_digital", "digitized"} & set(flags):
        confidence = Confidence.LOW      # DateTimeDigitized alone is usually a scan/edit date
    elif flags:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.VERY_HIGH if has_camera else Confidence.MEDIUM
    return DateCandidate(interval, Source.EXIF, Precision.DAY, confidence,
                         f"{raw} [{device_string(exif)}]", tuple(flags))
