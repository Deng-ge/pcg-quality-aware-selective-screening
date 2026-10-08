from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocess.feature_logmel import compute_logmel, save_logmel_png  # noqa: E402
from data_preprocess.preprocess import bandpass_filter, peak_normalize  # noqa: E402


def synthesize_pcg(sr: int = 2000, duration: float = 8.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    wav = np.zeros_like(t)

    bpm = 72.0
    period = 60.0 / bpm
    beat_times = np.arange(0.35, duration, period)

    for beat in beat_times:
        s1 = np.exp(-0.5 * ((t - beat) / 0.035) ** 2) * np.sin(2 * np.pi * 55.0 * (t - beat))
        s2_center = beat + 0.28
        s2 = np.exp(-0.5 * ((t - s2_center) / 0.026) ** 2) * np.sin(2 * np.pi * 75.0 * (t - s2_center))
        wav += 0.9 * s1 + 0.65 * s2

    wav += 0.015 * np.sin(2 * np.pi * 8.0 * t)
    wav += rng.normal(0.0, 0.01, size=wav.shape).astype(np.float32)
    wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.72
    return wav.astype(np.float32)


def save_synthetic_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 2000
    wav = synthesize_pcg(sr=sr)
    wavfile.write(path, sr, np.round(wav * 32767).astype(np.int16))


def draw_pipeline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "PCG wav",
        "Resample\nFilter\nNormalize",
        "Log-Mel\nfeatures",
        "ResNet +\nattention",
        "Quality\nscore",
        "Reject /\nreview",
    ]
    colors = ["#2f6f73", "#6b8e4e", "#b7791f", "#8a5a9e", "#3b6ea8", "#9b4d4d"]

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.axis("off")
    width = 1.35
    gap = 0.28
    y = 0.35
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = i * (width + gap)
        box = plt.Rectangle((x, y), width, 0.58, facecolor=color, edgecolor="white", linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + width / 2, y + 0.29, label, ha="center", va="center", color="white", fontsize=10)
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.84, y + 0.29),
                xytext=(x + width + gap * 0.12, y + 0.29),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#333333"},
            )
    ax.set_xlim(-0.1, len(labels) * (width + gap) - gap + 0.1)
    ax.set_ylim(0.1, 1.15)
    plt.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    wav_path = PROJECT_ROOT / "examples" / "synthetic_pcg.wav"
    save_synthetic_wav(wav_path)

    sr = 2000
    wav = synthesize_pcg(sr=sr)
    wav = peak_normalize(bandpass_filter(wav, sr))
    logmel = compute_logmel(wav, sr=sr)
    save_logmel_png(logmel, PROJECT_ROOT / "figures" / "logmel_example.png", title="Synthetic PCG Log-Mel example")
    draw_pipeline(PROJECT_ROOT / "figures" / "pipeline.png")
    print(f"Saved {wav_path}")
    print("Saved figures/pipeline.png and figures/logmel_example.png")


if __name__ == "__main__":
    main()

