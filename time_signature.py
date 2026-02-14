#!/usr/bin/env python3
"""
Time signature detection (Gainza & Coyle, AES 2007) adapted to accept MP3 input.

Key points:
- Uses a beat-synchronous STFT grid: window = 1/32 beat, hop = 1/64 beat
- Finds first note-like frame (to skip leading silence)
- Builds an Audio Similarity Matrix (ASM) using Euclidean distance between frames,
  then converts distance -> similarity so we can maximize a similarity score
- Scores candidate bar lengths from 2..12 beats using the paper's multi-resolution diagonal grouping
- Optionally searches for anacrusis (pickup) by shifting the ASM origin
- Maps beats-per-bar to a time signature label

Dependencies:
  pip install numpy scipy matplotlib
For audio I/O:
  pip install soundfile
For MP3 support:
  pip install librosa audioread
And make sure ffmpeg is installed on your system for robust MP3 decoding.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


# ---------------------------
# Utility
# ---------------------------

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(math.ceil(math.log2(n)))

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x, dtype=np.float64)))

def resample_if_needed(y: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    if sr_in == sr_out:
        return y, sr_in
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y_rs = signal.resample_poly(y, up, down).astype(np.float32)
    return y_rs, sr_out

def load_audio_any(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Robust loader that can handle MP3:
    1) Try soundfile (great for wav/flac/ogg; may or may not support mp3 depending on build)
    2) Fallback to librosa.load (commonly works for mp3 via audioread/ffmpeg)

    Returns:
      y: mono float32 waveform
      sr: sample rate (target_sr)
    """
    # Try soundfile first
    try:
        import soundfile as sf  # optional
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        y, sr = resample_if_needed(y, int(sr), target_sr)
        return y, sr
    except Exception:
        pass

    # Fallback: librosa (recommended for mp3)
    try:
        import librosa  # optional but recommended
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as e:
        raise RuntimeError(
            "Failed to load audio. For MP3, install ffmpeg and `pip install librosa audioread`.\n"
            "If you only have WAV/FLAC, `pip install soundfile` may be enough."
        ) from e


# ---------------------------
# Core steps (paper-inspired)
# ---------------------------

def compute_beat_sync_spectrogram(
    y: np.ndarray,
    sr: int,
    tempo_bpm: float,
    start_time: float = 0.0,
    duration: Optional[float] = 12.0,
    max_freq_hz: float = 4000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STFT magnitude spectrogram with:
      window = 1/32 beat, hop = 1/64 beat
    Returns:
      S: (freq_bins, frames) log-power magnitude (restricted to max_freq_hz)
      freqs: frequency axis (Hz)
    """
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be > 0")

    beat_dur = 60.0 / tempo_bpm
    win_length = max(16, int(round(sr * beat_dur / 32.0)))
    hop_length = max(8, int(round(sr * beat_dur / 64.0)))
    n_fft = next_pow2(win_length)

    # Crop excerpt
    start_samp = int(round(start_time * sr))
    if duration is None:
        y_seg = y[start_samp:]
    else:
        end_samp = int(round((start_time + duration) * sr))
        y_seg = y[start_samp:end_samp]

    if y_seg.size < win_length:
        raise ValueError("Audio segment too short for the chosen STFT window. Increase --dur or check tempo.")

    f, _, Zxx = signal.stft(
        y_seg,
        fs=sr,
        window="hann",
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx).astype(np.float32)

    # Restrict to low freqs for speed/robustness
    keep = f <= max_freq_hz
    f = f[keep]
    mag = mag[keep, :]

    # log-power
    S = np.log1p(mag * mag).astype(np.float32)
    return S, f.astype(np.float32)


def detect_first_note_frame(
    S: np.ndarray,
    freqs_hz: np.ndarray,
    ratio_threshold: float = 30.0,
    min_lowband_energy: float = 1e-4,
) -> int:
    """
    Find first frame likely containing a note:
    Compare low-band energy (<=3kHz) vs very-high-band energy (top of available spectrum).
    Returns frame index (or 0 if none).
    """
    if S.shape[1] == 0:
        return 0

    # Convert log-power back to power-ish
    P = np.expm1(S).astype(np.float32)

    low = freqs_hz <= 3000.0
    if not np.any(low):
        low = freqs_hz <= (0.25 * float(freqs_hz.max()))

    fmax = float(freqs_hz.max()) if freqs_hz.size else 0.0
    high = freqs_hz >= (0.80 * fmax) if fmax > 0 else np.zeros_like(freqs_hz, dtype=bool)
    if not np.any(high):
        high = ~low  # fallback if spectrum is tiny

    E1 = np.sum(P[low, :], axis=0)
    E2 = np.sum(P[high, :], axis=0) + 1e-12
    ratio = E1 / E2

    idx = np.where((ratio >= ratio_threshold) & (E1 >= min_lowband_energy))[0]
    return int(idx[0]) if idx.size else 0


def build_similarity_matrix(S: np.ndarray, sigma: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """
    Build similarity matrix between frames.
    Paper ASM uses Euclidean distance; we convert to similarity via exp(-dist/sigma)
    so that higher values indicate greater similarity (and can be maximized).
    """
    X = S.T.astype(np.float32)  # (T, D)
    T = X.shape[0]
    if T == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Normalize per frame to reduce loudness effects
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Squared Euclidean distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
    ss = np.sum(X * X, axis=1, keepdims=True)  # (T,1)
    G = X @ X.T  # (T,T)
    D2 = ss + ss.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    dist = np.sqrt(D2).astype(np.float32)

    # Robust sigma estimate if not provided
    if sigma is None:
        if T <= 2:
            sigma = 1.0
        else:
            rng = np.random.default_rng(seed)
            n_samples = min(50000, (T * (T - 1)) // 2)
            ii = rng.integers(0, T, size=n_samples, endpoint=False)
            jj = rng.integers(0, T, size=n_samples, endpoint=False)
            mask = ii != jj
            vals = dist[ii[mask], jj[mask]]
            sigma = float(np.median(vals)) if vals.size else 1.0
            sigma = max(sigma, 1e-6)

    sim = np.exp(-dist / float(sigma)).astype(np.float32)
    return sim


def score_bar_length(sim: np.ndarray, bar_frames: int, offset_frames: int = 0) -> float:
    """
    Multi-resolution scoring:
    - Take diagonals at offsets = bar_frames, 2*bar_frames, ...
    - Split each diagonal into complete segments (length bar_frames) and an incomplete tail
    - Compute RMS per segment
    - Weighted average across all segments (weight by segment length)
    Returns SM (higher is better).
    """
    if sim.size == 0:
        return float("-inf")

    if offset_frames > 0:
        if offset_frames >= sim.shape[0] - 2:
            return float("-inf")
        sim = sim[offset_frames:, offset_frames:]

    T = sim.shape[0]
    if bar_frames <= 1 or bar_frames >= T:
        return float("-inf")

    numerator = 0.0
    denom = 0.0

    lag = bar_frames
    while lag < T:
        d = np.diagonal(sim, offset=lag)
        if d.size == 0:
            break

        sc = d.size // bar_frames
        r = d.size - sc * bar_frames

        if sc > 0:
            seg = d[: sc * bar_frames].reshape(sc, bar_frames)
            # RMS per segment
            rms_vals = np.sqrt(np.mean(seg * seg, axis=1, dtype=np.float64))
            numerator += bar_frames * float(np.sum(rms_vals))
            denom += bar_frames * sc

        if r > 0:
            tail = d[sc * bar_frames :]
            numerator += r * rms(tail)
            denom += r

        lag += bar_frames

    return (numerator / denom) if denom > 0 else float("-inf")


_TIME_SIG_MAP: Dict[int, str] = {
    2: "2/2",
    3: "3/4",
    4: "4/4",
    5: "5/4",
    6: "6/8",
    7: "7/8",
    8: "8/8",
    9: "9/8",
    10: "10/8",
    11: "11/8",
    12: "12/8",
}


@dataclass
class DetectionResult:
    best_bar_frames: int
    best_beats_per_bar: float
    rounded_beats: int
    time_signature: str
    anacrusis_beats: int
    similarity_score: float
    tempo_bpm_in: float
    tempo_bpm_refined: float
    curve: List[Tuple[float, float]]  # (beats_per_bar, SM)


def detect_time_signature(
    audio_path: str,
    tempo_bpm: float,
    target_sr: int = 44100,
    start_time: float = 0.0,
    duration: Optional[float] = 12.0,
    min_beats: int = 2,
    max_beats: int = 12,
    frames_per_beat: int = 64,
    bar_step_frames: int = 1,
    max_bar_seconds: float = 3.5,
    anacrusis_search: bool = True,
    tie_tol: float = 0.01,
) -> DetectionResult:
    """
    Full pipeline. Tempo is provided (as assumed in the paper's setup).
    """
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be > 0")

    y, sr = load_audio_any(audio_path, target_sr)

    # Spectrogram on beat-synchronous grid
    S, freqs = compute_beat_sync_spectrogram(
        y, sr, tempo_bpm, start_time=start_time, duration=duration, max_freq_hz=4000.0
    )

    # Skip leading silence/noise by detecting first note-like frame
    first = detect_first_note_frame(S, freqs, ratio_threshold=30.0)
    if first > 0:
        S = S[:, first:]

    sim = build_similarity_matrix(S)

    beat_dur = 60.0 / tempo_bpm
    min_bar = min_beats * frames_per_beat
    max_bar = max_beats * frames_per_beat

    curve: List[Tuple[float, float]] = []
    scores: List[Tuple[int, float]] = []

    for bar_frames in range(min_bar, max_bar + 1, bar_step_frames):
        beats = bar_frames / float(frames_per_beat)

        # Bound bar length in seconds (keeps search reasonable on fast/slow tempos)
        if beats * beat_dur > max_bar_seconds:
            continue

        sm = score_bar_length(sim, bar_frames, offset_frames=0)
        curve.append((beats, sm))
        scores.append((bar_frames, sm))

    if not scores:
        raise RuntimeError("No valid bar-length candidates scored. Check --tempo, --dur, or max_bar_seconds.")

    # Choose best score; if near-tie, prefer shorter bar (helps avoid selecting multiples)
    max_sm = max(sm for _, sm in scores)
    good = [(bf, sm) for bf, sm in scores if sm >= max_sm * (1.0 - tie_tol)]
    best_bar_frames, best_sm = min(good, key=lambda x: x[0])

    # Anacrusis (pickup) search: shift in integer beats up to (bar-1) beats
    best_ana_beats = 0
    if anacrusis_search:
        rounded_beats_tmp = int(round(best_bar_frames / float(frames_per_beat)))
        max_ana = max(0, rounded_beats_tmp - 1)
        for ana_beats in range(0, max_ana + 1):
            off = ana_beats * frames_per_beat
            sm = score_bar_length(sim, best_bar_frames, offset_frames=off)
            if sm > best_sm:
                best_sm = sm
                best_ana_beats = ana_beats

    best_beats = best_bar_frames / float(frames_per_beat)
    rounded_beats = int(round(best_beats))
    rounded_beats = int(np.clip(rounded_beats, min_beats, max_beats))

    ts = _TIME_SIG_MAP.get(rounded_beats, f"{rounded_beats}/4")

    # Paper note: treat 2/2 and 8/8 as 4/4 by tempo scaling; we report 4/4
    if ts in ("2/2", "8/8"):
        ts = "4/4"

    # Tempo refinement: scale tempo so best_beats becomes an integer (useful if best_beats is fractional)
    tempo_refined = tempo_bpm * (rounded_beats / best_beats) if best_beats > 1e-9 else tempo_bpm

    return DetectionResult(
        best_bar_frames=best_bar_frames,
        best_beats_per_bar=best_beats,
        rounded_beats=rounded_beats,
        time_signature=ts,
        anacrusis_beats=best_ana_beats,
        similarity_score=best_sm,
        tempo_bpm_in=tempo_bpm,
        tempo_bpm_refined=float(tempo_refined),
        curve=curve,
    )


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Time signature detection (multi-resolution ASM), works with MP3/WAV/FLAC depending on decoders."
    )
    ap.add_argument("--audio", required=True, help="Path to audio file (mp3/wav/flac/ogg...).")
    ap.add_argument("--tempo", required=True, type=float, help="Tempo in BPM (required).")
    ap.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100).")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (seconds) for excerpt.")
    ap.add_argument("--dur", type=float, default=12.0, help="Duration (seconds) for excerpt.")
    ap.add_argument("--no-anacrusis", action="store_true", help="Disable anacrusis search.")
    ap.add_argument("--plot", action="store_true", help="Plot SM vs beats/bar.")
    args = ap.parse_args()

    res = detect_time_signature(
        audio_path=args.audio,
        tempo_bpm=args.tempo,
        target_sr=args.sr,
        start_time=args.start,
        duration=args.dur,
        anacrusis_search=not args.no_anacrusis,
    )

    print("=== Detection result ===")
    print(f"Time signature:       {res.time_signature}")
    print(f"Beats/bar (best B):   {res.best_beats_per_bar:.4f} (rounded -> {res.rounded_beats})")
    print(f"Anacrusis (beats):    {res.anacrusis_beats}")
    print(f"Similarity score SM:  {res.similarity_score:.6f}")
    print(f"Tempo input (BPM):    {res.tempo_bpm_in:.3f}")
    print(f"Tempo refined (BPM):  {res.tempo_bpm_refined:.3f}")

    if args.plot:
        import matplotlib.pyplot as plt
        xs = [b for (b, sm) in res.curve]
        ys = [sm for (b, sm) in res.curve]
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Beats per bar (B)")
        plt.ylabel("Similarity measure (SM)")
        plt.title("Multi-resolution ASM time signature detection curve")
        plt.show()


if __name__ == "__main__":
    main()
