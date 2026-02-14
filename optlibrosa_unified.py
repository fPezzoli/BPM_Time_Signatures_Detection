#!/usr/bin/env python3
"""
Compatibility wrapper for the "lightweight" BPM estimation.

This module contains abaseline librosa beat-track that reduces compute by using a smaller mel spectrogram and
re-using it to compute onset strength.

To keep BPM tracking implemented in ONE place, this module now delegates to:
  beat_tracking_unified.py

Exports:
  - estimate_bpm_light(path)  -> float BPM
  - estimate_bpm_normal(path) -> float BPM
  - estimate_bpm_light_from_path / estimate_bpm_normal_from_path (full BPMResult)
"""

from __future__ import annotations

from typing import Any

from beat_tracking_unified import (
    BPMResult,
    estimate_bpm_light_from_path,
    estimate_bpm_normal_from_path,
)

__all__ = [
    "BPMResult",
    "estimate_bpm_light_from_path",
    "estimate_bpm_normal_from_path",
    "estimate_bpm_light",
    "estimate_bpm_normal",
]


def estimate_bpm_light(path: str, **kwargs: Any) -> float:
    """Return BPM using the lightweight pipeline."""
    return float(estimate_bpm_light_from_path(path, **kwargs).bpm)


def estimate_bpm_normal(path: str, **kwargs: Any) -> float:
    """Return BPM using the normal onset->tempo pipeline."""
    return float(estimate_bpm_normal_from_path(path, **kwargs).bpm)
