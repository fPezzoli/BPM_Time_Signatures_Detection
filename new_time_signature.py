#!/usr/bin/env python3
"""
Time signature tracking over time using a windowed version of the
multi-resolution Audio Similarity Matrix (ASM) approach (Gainza & Coyle, AES 2007).

- MP3 input supported (soundfile if possible, else librosa/audioread).
- Beat-synchronous STFT grid: window = 1/32 beat, hop = 1/64 beat.
- Sliding windows over the song to detect time-signature changes.
- Viterbi smoothing to avoid jitter.
- Outputs segments with start/end times and time signature.

Dependencies:
  pip install numpy scipy matplotlib
Audio I/O:
  pip install soundfile
MP3 support:
  pip install librosa audioread
Also ensure ffmpeg is installed for robust mp3 decoding.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.special import logsumexp


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
    1) Try soundfile (fast; may or may not support mp3 depending on libsndfile build)
    2) Fallback to librosa.load (commonly works for mp3 via audioread/ffmpeg)
    """
    # Try soundfile first
    try:
        import soundfile as sf
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
        import librosa
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as e:
        raise RuntimeError(
            "Failed to load audio. For MP3, install ffmpeg and `pip install librosa audioread`.\n"
            "If using WAV/FLAC, `pip install soundfile` may be enough."
        ) from e


# ---------------------------
# Paper-inspired components
# ---------------------------

def compute_beat_sync_spectrogram_full(
    y: np.ndarray,
    sr: int,
    tempo_bpm: float,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    max_freq_hz: float = 4000.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute log-power STFT magnitude spectrogram with:
      window = 1/32 beat, hop = 1/64 beat
    Returns:
      S: (freq_bins, frames) log-power
      freqs_hz: (freq_bins,)
      hop_length_samples: int
    """
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be > 0")

    beat_dur = 60.0 / tempo_bpm
    win_length = max(16, int(round(sr * beat_dur / 32.0)))
    hop_length = max(8, int(round(sr * beat_dur / 64.0)))
    n_fft = next_pow2(win_length)

    start_samp = int(round(start_time * sr))
    if duration is None:
        y_seg = y[start_samp:]
    else:
        end_samp = int(round((start_time + duration) * sr))
        y_seg = y[start_samp:end_samp]

    if y_seg.size < win_length:
        raise ValueError("Audio too short for the chosen STFT window. Increase duration or check tempo.")

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

    keep = f <= max_freq_hz
    f = f[keep]
    mag = mag[keep, :]

    S = np.log1p(mag * mag).astype(np.float32)
    return S, f.astype(np.float32), hop_length


def detect_first_note_frame(
    S: np.ndarray,
    freqs_hz: np.ndarray,
    ratio_threshold: float = 30.0,
    min_lowband_energy: float = 1e-4,
) -> int:
    """
    Heuristic to skip leading silence/noise:
    compare low-band energy (<=3kHz) against very-high band (top of available spectrum).
    Returns frame index (or 0 if none).
    """
    if S.shape[1] == 0:
        return 0

    P = np.expm1(S).astype(np.float32)  # approx power

    low = freqs_hz <= 3000.0
    if not np.any(low):
        low = freqs_hz <= (0.25 * float(freqs_hz.max()))

    fmax = float(freqs_hz.max()) if freqs_hz.size else 0.0
    high = freqs_hz >= (0.80 * fmax) if fmax > 0 else np.zeros_like(freqs_hz, dtype=bool)
    if not np.any(high):
        high = ~low

    E1 = np.sum(P[low, :], axis=0)
    E2 = np.sum(P[high, :], axis=0) + 1e-12
    ratio = E1 / E2

    idx = np.where((ratio >= ratio_threshold) & (E1 >= min_lowband_energy))[0]
    return int(idx[0]) if idx.size else 0


def build_similarity_matrix(S_win: np.ndarray, sigma: Optional[float] = None, seed: int = 0) -> np.ndarray:
    """
    ASM distance is Euclidean; we convert to similarity:
      sim = exp(-dist/sigma)
    """
    X = S_win.T.astype(np.float32)  # (T, D)
    T = X.shape[0]
    if T == 0:
        return np.zeros((0, 0), dtype=np.float32)

    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    ss = np.sum(X * X, axis=1, keepdims=True)
    G = X @ X.T
    D2 = ss + ss.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    dist = np.sqrt(D2).astype(np.float32)

    if sigma is None:
        if T <= 2:
            sigma = 1.0
        else:
            rng = np.random.default_rng(seed)
            n_samples = min(20000, (T * (T - 1)) // 2)
            ii = rng.integers(0, T, size=n_samples, endpoint=False)
            jj = rng.integers(0, T, size=n_samples, endpoint=False)
            mask = ii != jj
            vals = dist[ii[mask], jj[mask]]
            sigma = float(np.median(vals)) if vals.size else 1.0
            sigma = max(sigma, 1e-6)

    return np.exp(-dist / float(sigma)).astype(np.float32)


def score_bar_length(sim: np.ndarray, bar_frames: int) -> float:
    """
    Multi-resolution diagonal grouping score (higher is better).
    """
    if sim.size == 0:
        return float("-inf")

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
            seg_rms = np.sqrt(np.mean(seg * seg, axis=1, dtype=np.float64))
            numerator += bar_frames * float(np.sum(seg_rms))
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

def beats_to_label(beats_per_bar: int) -> str:
    ts = _TIME_SIG_MAP.get(beats_per_bar, f"{beats_per_bar}/4")
    # paper notes often equate 2/2 and 8/8 to 4/4 by tempo scaling; report 4/4
    if ts in ("2/2", "8/8"):
        return "4/4"
    return ts


# ---------------------------
# Windowed tracking + smoothing
# ---------------------------

@dataclass
class WindowResult:
    center_time: float
    scores_by_beats: Dict[int, float]  # beats -> SM score
    raw_beats: int

@dataclass
class Segment:
    start: float
    end: float
    beats_per_bar: int
    time_signature: str

def viterbi_smooth(
    windows: List[WindowResult],
    states: List[int],
    transition_penalty: float = 0.7,
    emission_temperature: float = 0.02,
) -> List[int]:
    """
    Viterbi decode over integer beats-per-bar states.
    Emissions derived from per-window SM scores using a softmax temperature.
    Transition penalty discourages rapid switching; larger => fewer changes.

    transition cost = -transition_penalty * abs(b_i - b_{i-1})
    """
    N = len(windows)
    S = len(states)
    if N == 0:
        return []

    # Build emission log-probabilities: shape (N, S)
    E = np.full((N, S), -np.inf, dtype=np.float64)
    for t, w in enumerate(windows):
        sm = np.array([w.scores_by_beats.get(b, -1e9) for b in states], dtype=np.float64)
        # softmax over SM/temperature -> log probs
        logits = sm / max(emission_temperature, 1e-9)
        E[t] = logits - logsumexp(logits)

    # DP
    dp = np.full((N, S), -np.inf, dtype=np.float64)
    bp = np.zeros((N, S), dtype=np.int32)

    dp[0] = E[0]
    for t in range(1, N):
        for j, bj in enumerate(states):
            trans = np.array(
                [dp[t - 1, i] - transition_penalty * abs(bj - states[i]) for i in range(S)],
                dtype=np.float64,
            )
            best_i = int(np.argmax(trans))
            dp[t, j] = E[t, j] + trans[best_i]
            bp[t, j] = best_i

    path_idx = [int(np.argmax(dp[-1]))]
    for t in range(N - 1, 0, -1):
        path_idx.append(int(bp[t, path_idx[-1]]))
    path_idx.reverse()

    return [states[i] for i in path_idx]


def detect_time_signature_changes(
    audio_path: str,
    tempo_bpm: float,
    target_sr: int = 44100,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    frames_per_beat: int = 64,
    # Windowing in BEATS (works well because the STFT grid is beat-synchronous)
    window_beats: int = 32,
    hop_beats: int = 8,
    # Candidate integer meters
    min_beats_per_bar: int = 2,
    max_beats_per_bar: int = 12,
    # Speed/robustness controls
    max_bar_seconds: float = 3.5,
    allow_small_frame_jitter: int = 2,  # evaluate +/- this many frames around exact b*64
    # Smoothing
    transition_penalty: float = 0.7,
    emission_temperature: float = 0.02,
    # Segment cleanup
    min_segment_seconds: float = 0.0,
) -> Tuple[List[Segment], List[WindowResult]]:
    """
    Returns:
      segments: merged time-signature segments
      windows: raw per-window results (useful for plotting/debugging)
    """
    if tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be > 0")

    y, sr = load_audio_any(audio_path, target_sr)

    S, freqs, hop_len = compute_beat_sync_spectrogram_full(
        y, sr, tempo_bpm, start_time=start_time, duration=duration, max_freq_hz=4000.0
    )

    # First-note trim
    first = detect_first_note_frame(S, freqs, ratio_threshold=30.0)
    if first > 0:
        S = S[:, first:]

    hop_sec = hop_len / float(sr)
    analysis_start_sec = start_time + first * hop_sec

    T_frames = S.shape[1]
    if T_frames < window_beats * frames_per_beat:
        raise RuntimeError("Track (or selected duration) too short for chosen window_beats.")

    beat_dur = 60.0 / tempo_bpm
    states = list(range(min_beats_per_bar, max_beats_per_bar + 1))

    window_frames = window_beats * frames_per_beat
    hop_frames = hop_beats * frames_per_beat

    windows: List[WindowResult] = []

    for w0 in range(0, T_frames - window_frames + 1, hop_frames):
        Sw = S[:, w0 : w0 + window_frames]
        sim = build_similarity_matrix(Sw)

        scores_by_beats: Dict[int, float] = {}
        for b in states:
            # skip if bar duration too long in seconds (optional bound)
            if (b * beat_dur) > max_bar_seconds:
                scores_by_beats[b] = float("-inf")
                continue

            base = b * frames_per_beat
            best_sm = float("-inf")
            for delta in range(-allow_small_frame_jitter, allow_small_frame_jitter + 1):
                bf = base + delta
                if 2 <= bf < sim.shape[0]:
                    best_sm = max(best_sm, score_bar_length(sim, bf))
            scores_by_beats[b] = best_sm

        raw_beats = max(scores_by_beats.items(), key=lambda kv: kv[1])[0]
        center_time = analysis_start_sec + (w0 + window_frames / 2.0) * hop_sec
        windows.append(WindowResult(center_time=center_time, scores_by_beats=scores_by_beats, raw_beats=raw_beats))

    # Smooth via Viterbi
    smooth_beats = viterbi_smooth(
        windows,
        states=states,
        transition_penalty=transition_penalty,
        emission_temperature=emission_temperature,
    )

    # Convert to segments using midpoints between window centers
    centers = [w.center_time for w in windows]
    segments: List[Segment] = []
    if not centers:
        return segments, windows

    seg_start = analysis_start_sec
    current = smooth_beats[0]

    for i in range(1, len(smooth_beats)):
        if smooth_beats[i] != current:
            boundary = 0.5 * (centers[i - 1] + centers[i])
            segments.append(Segment(seg_start, boundary, current, beats_to_label(current)))
            seg_start = boundary
            current = smooth_beats[i]

    # End of last segment: approx end of analyzed audio
    analysis_end_sec = analysis_start_sec + T_frames * hop_sec
    segments.append(Segment(seg_start, analysis_end_sec, current, beats_to_label(current)))

    # Optionally merge tiny segments
    if min_segment_seconds > 0 and len(segments) > 1:
        merged: List[Segment] = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            if (seg.end - seg.start) >= min_segment_seconds or len(merged) == 0:
                merged.append(seg)
                i += 1
                continue

            # Merge short segment into previous (simple policy)
            prev = merged[-1]
            merged[-1] = Segment(prev.start, seg.end, prev.beats_per_bar, prev.time_signature)
            i += 1
        segments = merged

    return segments, windows


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Time signature tracking over time (windowed ASM), MP3 supported.")
    ap.add_argument("--audio", required=True, help="Path to audio file (mp3/wav/flac/ogg...).")
    ap.add_argument("--tempo", required=True, type=float, help="Tempo in BPM (assumed roughly stable).")
    ap.add_argument("--sr", type=int, default=44100, help="Target sample rate (default 44100).")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    ap.add_argument("--dur", type=float, default=-1.0, help="Duration in seconds (default: full track).")
    ap.add_argument("--window-beats", type=int, default=32, help="Window length in beats (default 32).")
    ap.add_argument("--hop-beats", type=int, default=8, help="Hop length in beats (default 8).")
    ap.add_argument("--transition-penalty", type=float, default=0.7, help="Penalty for changing meter (default 0.7).")
    ap.add_argument("--temp", type=float, default=0.02, help="Emission softmax temperature (default 0.02).")
    ap.add_argument("--min-seg-sec", type=float, default=0.0, help="Merge segments shorter than this many seconds.")
    ap.add_argument("--plot", action="store_true", help="Plot beats/bar over time.")
    args = ap.parse_args()

    dur = None if args.dur is None or args.dur < 0 else float(args.dur)

    segments, windows = detect_time_signature_changes(
        audio_path=args.audio,
        tempo_bpm=args.tempo,
        target_sr=args.sr,
        start_time=args.start,
        duration=dur,
        window_beats=args.window_beats,
        hop_beats=args.hop_beats,
        transition_penalty=args.transition_penalty,
        emission_temperature=args.temp,
        min_segment_seconds=args.min_seg_sec,
    )

    print("=== Time signature segments ===")
    for s in segments:
        print(f"{s.start:8.2f}s  -> {s.end:8.2f}s   {s.time_signature}  (beats/bar={s.beats_per_bar})")

    if args.plot:
        import matplotlib.pyplot as plt
        times = [w.center_time for w in windows]
        raw = [w.raw_beats for w in windows]

        # recompute smoothed for plotting (same as inside)
        states = list(range(2, 13))
        smooth = viterbi_smooth(
            windows,
            states=states,
            transition_penalty=args.transition_penalty,
            emission_temperature=args.temp,
        )

        plt.figure()
        plt.plot(times, raw, label="raw")
        plt.plot(times, smooth, label="smoothed")
        plt.xlabel("Time (s)")
        plt.ylabel("Beats per bar")
        plt.title("Time signature tracking over time")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
