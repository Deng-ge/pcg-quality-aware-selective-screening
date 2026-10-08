from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt


def _pcm_to_float32(wav: np.ndarray) -> np.ndarray:
    if np.issubdtype(wav.dtype, np.integer):
        info = np.iinfo(wav.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        wav = wav.astype(np.float32) / scale
    else:
        wav = wav.astype(np.float32)
    return wav


def load_wav(path: str | Path, target_sr: int = 2000) -> Tuple[np.ndarray, int]:
    """Load a wav file, convert to mono float32, and resample if needed."""
    sr, wav = wavfile.read(str(path))
    wav = _pcm_to_float32(wav)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    if sr != target_sr:
        factor = gcd(int(sr), int(target_sr))
        up = int(target_sr // factor)
        down = int(sr // factor)
        wav = resample_poly(wav, up, down).astype(np.float32)
        sr = target_sr

    return wav.astype(np.float32), int(sr)


def bandpass_filter(
    wav: np.ndarray,
    sr: int,
    low_hz: float = 20.0,
    high_hz: float = 800.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter for the main PCG band."""
    nyquist = 0.5 * sr
    high_hz = min(float(high_hz), nyquist * 0.98)
    low_hz = max(float(low_hz), 0.1)

    if low_hz >= high_hz:
        return wav.astype(np.float32)

    sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    if len(wav) < 3 * max(1, order):
        return wav.astype(np.float32)
    return sosfiltfilt(sos, wav).astype(np.float32)


def peak_normalize(wav: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < eps:
        return wav.astype(np.float32)
    return (wav / (peak + eps)).astype(np.float32)


def center_crop_or_pad(wav: np.ndarray, target_len: int) -> np.ndarray:
    """Return a fixed-length center crop, zero-padding if the signal is short."""
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    if len(wav) == target_len:
        return wav.astype(np.float32)

    if len(wav) < target_len:
        pad_total = target_len - len(wav)
        left = pad_total // 2
        right = pad_total - left
        return np.pad(wav, (left, right), mode="constant").astype(np.float32)

    start = (len(wav) - target_len) // 2
    return wav[start : start + target_len].astype(np.float32)


def sliding_windows(wav: np.ndarray, window_len: int, step_len: int) -> np.ndarray:
    """Split a waveform into fixed-length windows."""
    window_len = int(window_len)
    step_len = int(step_len)
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_len and step_len must be positive")

    if len(wav) <= window_len:
        return center_crop_or_pad(wav, window_len)[None, :]

    chunks = []
    for start in range(0, len(wav) - window_len + 1, step_len):
        chunks.append(wav[start : start + window_len])

    if not chunks:
        chunks.append(center_crop_or_pad(wav, window_len))
    return np.stack(chunks).astype(np.float32)

