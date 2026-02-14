#!/usr/bin/env python3
"""
Beat / BPM tracking utilities.

This file contains:
- The "normal" BPM estimation flow (onset envelope -> tempo)
- A "lightweight" BPM estimation flow (smaller mel spectrogram -> onset envelope -> tempo).

It also retains the original DP beat-tracking routine (Columbia-style) via MusicToolBox.beat_tracking_dp()
for users who want beat positions, though the main intended output for downstream time-signature detection is BPM.

MP3 input is supported via a robust loader:
- Try soundfile first (mp3 support depends on your libsndfile build)
- Fall back to librosa.load (typically uses audioread/ffmpeg)

Dependencies:
  pip install numpy librosa scipy soundfile
For robust MP3 decoding:
  pip install audioread
and ensure ffmpeg is installed on your system.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import signal

try:
    import librosa
except ImportError as e:
    raise ImportError("This module requires librosa. Install with: pip install librosa") from e


# ---------------------------
# Audio loading
# ---------------------------

def _resample_if_needed(y: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False), sr_in
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y_rs = signal.resample_poly(y, up, down).astype(np.float32)
    return y_rs, sr_out


def load_audio_any(path: str, target_sr: int, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio from disk

    Returns:
      y: float32 mono waveform (if mono=True)
      sr: target_sr
    """
    # 1) Try soundfile
    try:
        import soundfile as sf  # optional dependency
        y, sr = sf.read(path, always_2d=False)
        if mono and getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32, copy=False)
        y, sr = _resample_if_needed(y, int(sr), int(target_sr))
        return y, sr
    except Exception:
        pass

    # 2) Fallback: librosa (recommended for mp3)
    try:
        y, sr = librosa.load(path, sr=int(target_sr), mono=mono)
        return y.astype(np.float32, copy=False), int(sr)
    except Exception as e:
        raise RuntimeError(
            "Failed to load audio. For MP3, install ffmpeg and `pip install librosa audioread`.\n"
            "If using WAV/FLAC, `pip install soundfile` is usually sufficient."
        ) from e


# ---------------------------
# Tempo helpers (librosa API compatibility)
# ---------------------------

def _tempo_from_onset_envelope(
    onset_env: np.ndarray,
    sr: int,
    hop_length: int,
    start_bpm: float = 120.0,
) -> float:
    """
    Robustly compute global tempo (BPM) from onset envelope across librosa versions.
    """
    onset_env = np.asarray(onset_env, dtype=np.float32)

    # Preferred (classic)
    if hasattr(librosa, "beat") and hasattr(librosa.beat, "tempo"):
        bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, start_bpm=start_bpm)
        return float(np.atleast_1d(bpm)[0])

    # Some versions expose tempo in librosa.feature
    if hasattr(librosa, "feature") and hasattr(librosa.feature, "tempo"):
        bpm = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, start_bpm=start_bpm)
        return float(np.atleast_1d(bpm)[0])

    # librosa>=0.10: librosa.feature.rhythm.tempo
    if hasattr(librosa, "feature") and hasattr(librosa.feature, "rhythm") and hasattr(librosa.feature.rhythm, "tempo"):
        bpm = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, start_bpm=start_bpm)
        return float(np.atleast_1d(bpm)[0])

    raise RuntimeError("Could not find a tempo() function in the installed librosa version.")


# ---------------------------
# Normal approach
# ---------------------------

class MusicToolBox:
    """
    Minimal wrapper around onset detection/strength + tempo estimation,
    plus a DP beat-tracking routine.
    """

    def __init__(self, y: np.ndarray, sr: int, hop_length: int = 512):
        self.signal = np.asarray(y, dtype=np.float32)
        self.sr = int(sr)
        self.hop_length = int(hop_length)

    def onsets_detection(self) -> np.ndarray:
        return librosa.onset.onset_detect(y=self.signal, sr=self.sr, hop_length=self.hop_length)

    def onset_strength_curve(self) -> np.ndarray:
        return librosa.onset.onset_strength(y=self.signal, sr=self.sr, hop_length=self.hop_length).astype(np.float32)

    def estimate_tempo_in_bpm(self, onset_envelope: np.ndarray, start_bpm: float = 120.0) -> float:
        return _tempo_from_onset_envelope(onset_envelope, sr=self.sr, hop_length=self.hop_length, start_bpm=start_bpm)

    def estimate_tempo_in_frames(self, tempo_bpm: float) -> int:
        # frames per beat = (60/bpm) * (sr/hop)
        tempo_frames = (60.0 / float(tempo_bpm)) * (self.sr / self.hop_length)
        return int(round(tempo_frames))

    def beat_tracking_dp(self, onset_envelope: np.ndarray, alpha: float, tempo_frames: int) -> np.ndarray:
        """
        Dynamic-programming beat tracking.

        Args:
          onset_envelope: onset strength curve O(t)
          alpha: transition cost weight
          tempo_frames: global tempo in frames per beat

        Returns:
          beat frame indices (int array)
        """
        O = np.asarray(onset_envelope, dtype=np.float32)
        T = int(O.shape[0])
        if T == 0:
            return np.array([], dtype=int)

        prev_beat = np.full(T, -1, dtype=int)
        c_score = O.astype(np.float64).copy()

        tempo_min = -int(round(2 * tempo_frames))
        tempo_max = -int(round(tempo_frames / 2))
        tempo_range = np.arange(tempo_min, tempo_max + 1, dtype=int)

        # Transition costs
        transition_f = np.empty_like(tempo_range, dtype=np.float64)
        for k, p in enumerate(tempo_range):
            delta = -p
            ratio = delta / float(tempo_frames)
            transition_f[k] = -float(alpha) * (math.log(ratio) ** 2)

        start_i = -int(tempo_range.min())

        for i in range(start_i, T):
            best_value = -float("inf")
            best_tau = -1

            for k, p in enumerate(tempo_range):
                tau = i + int(p)
                if tau < 0 or tau >= T:
                    continue
                candidate = transition_f[k] + c_score[tau]
                if candidate > best_value:
                    best_value = candidate
                    best_tau = tau

            c_score[i] = float(O[i]) + best_value
            prev_beat[i] = best_tau

        end_time = int(np.argmax(c_score))
        beats = [end_time]
        while prev_beat[beats[0]] != -1:
            beats.insert(0, int(prev_beat[beats[0]]))
        return np.asarray(beats, dtype=int)


@dataclass
class BPMResult:
    bpm: float
    sr: int
    hop_length: int
    onset_envelope: np.ndarray


def estimate_bpm_normal_from_path(
    path: str,
    sr: int = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0,
) -> BPMResult:
    """
    "Normal" BPM estimation:
      y -> onset_strength -> tempo
    """
    y, sr = load_audio_any(path, target_sr=sr, mono=True)
    toolbox = MusicToolBox(y, sr=sr, hop_length=hop_length)
    onset_env = toolbox.onset_strength_curve()
    bpm = toolbox.estimate_tempo_in_bpm(onset_env, start_bpm=start_bpm)
    return BPMResult(bpm=float(bpm), sr=int(sr), hop_length=int(hop_length), onset_envelope=onset_env)


# ---------------------------
# Lightweight approach
# ---------------------------

def estimate_bpm_light_from_path(
    path: str,
    sr: int = 11025,
    hop_length: int = 1024,
    n_fft: int = 1024,
    n_mels: int = 64,
    start_bpm: float = 120.0,
) -> BPMResult:
    """
    "Light" BPM estimation:
      y -> small mel spectrogram -> onset_strength(S=...) -> tempo
    """
    y, sr = load_audio_any(path, target_sr=sr, mono=True)

    # 1) Smaller mel spectrogram (fewer bins, lower SR, larger hop => faster)
    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        power=2.0,
    ).astype(np.float32, copy=False)

    # 2) Convert to log-power (onset_strength expects log-ish S if given)
    M_db = librosa.power_to_db(M, ref=np.max).astype(np.float32, copy=False)

    # 3) Onset envelope from precomputed spectrogram (avoids recomputation)
    onset_env = librosa.onset.onset_strength(
        S=M_db,
        sr=sr,
        hop_length=int(hop_length),
        aggregate=np.median,
    ).astype(np.float32, copy=False)

    if not np.any(onset_env):
        return BPMResult(bpm=0.0, sr=int(sr), hop_length=int(hop_length), onset_envelope=onset_env)

    bpm = _tempo_from_onset_envelope(onset_env, sr=sr, hop_length=int(hop_length), start_bpm=start_bpm)
    return BPMResult(bpm=float(bpm), sr=int(sr), hop_length=int(hop_length), onset_envelope=onset_env)


def estimate_bpm_from_path(
    path: str,
    mode: Literal["normal", "light"] = "normal",
) -> BPMResult:
    """
    Convenience wrapper.
    """
    if mode == "normal":
        return estimate_bpm_normal_from_path(path)
    if mode == "light":
        return estimate_bpm_light_from_path(path)
    raise ValueError("mode must be 'normal' or 'light'")


# ---------------------------
# CLI (optional)
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate BPM from an audio file (normal or lightweight mode).")
    ap.add_argument("input", help="Audio file path (mp3/wav/flac/ogg...)")
    ap.add_argument("--mode", choices=["normal", "light"], default="normal")
    args = ap.parse_args()

    res = estimate_bpm_from_path(args.input, mode=args.mode)
    print(f"BPM ({args.mode}): {res.bpm:.2f}")


if __name__ == "__main__":
    main()
