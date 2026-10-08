from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 64,
    f_min: float = 20.0,
    f_max: float = 800.0,
) -> np.ndarray:
    """Create a triangular mel filter bank."""
    f_max = min(float(f_max), sr / 2.0)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        right = min(right, n_fft // 2)

        for j in range(left, min(center, bank.shape[1])):
            bank[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, bank.shape[1])):
            bank[i - 1, j] = (right - j) / max(right - center, 1)

    return bank


def compute_logmel(
    wav: np.ndarray,
    sr: int = 2000,
    n_mels: int = 64,
    n_fft: int = 256,
    hop_length: int = 80,
    f_min: float = 20.0,
    f_max: float = 800.0,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a normalized Log-Mel spectrogram."""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)), mode="constant")

    _, _, zxx = stft(
        wav,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx).astype(np.float32) ** 2
    mel_bank = mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    mel_spec = np.maximum(mel_bank @ power, 1e-10)
    logmel = np.log(mel_spec).astype(np.float32)

    if normalize:
        logmel = (logmel - float(logmel.mean())) / (float(logmel.std()) + 1e-6)
    return logmel.astype(np.float32)


def save_logmel_png(logmel: np.ndarray, output_path: str | Path, title: str = "Log-Mel spectrogram") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 3.2))
    plt.imshow(logmel, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="normalized log energy")
    plt.xlabel("frame")
    plt.ylabel("mel bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

