#!/usr/bin/env python3
"""
End-to-end CLI:
1) Estimate BPM (normal or light)
2) Run time signature detection (standard or varying)
3) Optionally plot the time signature result

Parameters:
  1) input (audio path)
  2) bpm mode: normal | light
  3) time signature mode: standard | varying
  4) plot (flag)

Examples:
  python bpm_ts_cli.py song.mp3 --bpm-mode normal --ts-mode standard --plot
  python bpm_ts_cli.py song.mp3 --bpm-mode light  --ts-mode varying  --plot
"""

from __future__ import annotations

import argparse

import numpy as np

# BPM estimation
from beat_tracking_unified import estimate_bpm_normal_from_path
from optlibrosa_unified import estimate_bpm_light_from_path

# Time signature detectors
import time_signature
import new_time_signature


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute BPM then time signature (standard or varying).")
    ap.add_argument("input", help="Input audio file")

    ap.add_argument(
        "--bpm-mode",
        choices=["normal", "light"],
        default="normal",
        help="BPM estimation mode: normal (beat_tracking) or light (optlibrosa lightweight).",
    )
    ap.add_argument(
        "--ts-mode",
        choices=["standard", "varying"],
        default="standard",
        help="Time signature mode: standard (single TS) or varying (segments over time).",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="If set, show a plot of the time signature output.",
    )
    args = ap.parse_args()

    # --- BPM ---
    if args.bpm_mode == "normal":
        bpm_res = estimate_bpm_normal_from_path(args.input)
    else:
        bpm_res = estimate_bpm_light_from_path(args.input)

    bpm = float(bpm_res.bpm)
    print(f"BPM ({args.bpm_mode}): {bpm:.2f}")

    if bpm <= 0:
        raise SystemExit("BPM estimation failed (<=0). Try the other bpm mode or a different file.")

    # --- Time Signature ---
    if args.ts_mode == "standard":
        ts_res = time_signature.detect_time_signature(audio_path=args.input, tempo_bpm=bpm)

        print(f"Time signature (standard): {ts_res.time_signature}")

        if args.plot:
            import matplotlib.pyplot as plt

            xs = [b for (b, sm) in ts_res.curve]
            ys = [sm for (b, sm) in ts_res.curve]

            plt.figure()
            plt.plot(xs, ys)
            plt.xlabel("Beats per bar (B)")
            plt.ylabel("Similarity measure (SM)")
            plt.title(f"Standard TS detection curve (pred={ts_res.time_signature})")
            plt.show()

    else:
        segments, windows = new_time_signature.detect_time_signature_changes(
            audio_path=args.input,
            tempo_bpm=bpm,
        )

        if not segments:
            print("Time signature (varying): <no segments detected>")
            return

        print("Time signature (varying) segments:")
        for s in segments:
            print(f"{s.start:8.2f}s -> {s.end:8.2f}s   {s.time_signature} (beats/bar={s.beats_per_bar})")

        if args.plot:
            import matplotlib.pyplot as plt

            times = [w.center_time for w in windows]
            raw = [w.raw_beats for w in windows]

            # Smooth again for plotting using the same defaults as the module
            states = list(range(2, 13))
            smooth = new_time_signature.viterbi_smooth(
                windows,
                states=states,
                transition_penalty=0.7,
                emission_temperature=0.02,
            )

            plt.figure()
            plt.plot(times, raw, label="raw")
            plt.plot(times, smooth, label="smoothed")
            plt.xlabel("Time (s)")
            plt.ylabel("Beats per bar")
            plt.title("Varying TS tracking over time")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    import sys

    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)  # Ctrl+C
    except FileNotFoundError as e:
        print(f"Error: file not found: {e}")
        sys.exit(2)
    except ValueError as e:
        print(f"Error: invalid value: {e}")
        sys.exit(2)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

