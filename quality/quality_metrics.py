from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.signal import welch


def _band_power(wav: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    if len(wav) < 4 or high_hz <= low_hz:
        return 0.0

    nperseg = min(1024, len(wav))
    freqs, pxx = welch(wav, fs=sr, nperseg=nperseg)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(pxx[mask], freqs[mask]))


def compute_quality_metrics(wav: np.ndarray, sr: int, cfg: Dict | None = None) -> Dict[str, float]:
    """Compute simple, interpretable PCG quality metrics."""
    cfg = cfg or {}
    x = np.asarray(wav, dtype=np.float32)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "crest_factor": 0.0,
            "clipping_ratio": 1.0,
            "flatline_ratio": 1.0,
            "lowfreq_ratio": 1.0,
            "highfreq_ratio": 1.0,
            "heart_band_ratio": 0.0,
            "quality_score": 0.0,
        }

    duration = float(len(x) / sr)
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    crest = float(peak / (rms + 1e-8))

    clipping_ratio = float(np.mean(np.abs(x) >= 0.98))
    if len(x) > 1:
        flatline_ratio = float(np.mean(np.abs(np.diff(x)) < 1e-6))
    else:
        flatline_ratio = 1.0

    nyquist = sr / 2.0
    total = _band_power(x, sr, 0.5, min(900.0, nyquist * 0.98)) + 1e-12
    lowfreq = _band_power(x, sr, 0.5, 20.0) / total
    highfreq = _band_power(x, sr, 400.0, min(900.0, nyquist * 0.98)) / total
    heart_band = _band_power(x, sr, 20.0, 200.0) / total

    min_rms = float(cfg.get("min_rms", 0.015))
    max_low = float(cfg.get("max_lowfreq_ratio", 0.25))
    max_high = float(cfg.get("max_highfreq_ratio", 0.35))
    max_clip = float(cfg.get("max_clipping_ratio", 0.005))
    max_flat = float(cfg.get("max_flatline_ratio", 0.05))

    score = 1.0
    if duration < 3.0:
        score -= min(0.25, (3.0 - duration) / 3.0 * 0.25)
    if rms < min_rms:
        score -= min(0.35, (min_rms - rms) / max(min_rms, 1e-8) * 0.35)
    if clipping_ratio > max_clip:
        score -= min(0.30, (clipping_ratio - max_clip) * 12.0)
    if flatline_ratio > max_flat:
        score -= min(0.25, (flatline_ratio - max_flat) * 2.0)
    if lowfreq > max_low:
        score -= min(0.25, (lowfreq - max_low) * 0.8)
    if highfreq > max_high:
        score -= min(0.25, (highfreq - max_high) * 0.8)

    return {
        "duration_seconds": duration,
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "clipping_ratio": clipping_ratio,
        "flatline_ratio": flatline_ratio,
        "lowfreq_ratio": float(lowfreq),
        "highfreq_ratio": float(highfreq),
        "heart_band_ratio": float(heart_band),
        "quality_score": float(np.clip(score, 0.0, 1.0)),
    }


def describe_quality_issues(metrics: Dict[str, float], cfg: Dict | None = None) -> List[str]:
    cfg = cfg or {}
    issues: List[str] = []

    if metrics["duration_seconds"] < 3.0:
        issues.append("recording too short")
    if metrics["rms"] < float(cfg.get("min_rms", 0.015)):
        issues.append("low signal amplitude")
    if metrics["clipping_ratio"] > float(cfg.get("max_clipping_ratio", 0.005)):
        issues.append("possible clipping")
    if metrics["flatline_ratio"] > float(cfg.get("max_flatline_ratio", 0.05)):
        issues.append("flat or repeated samples")
    if metrics["lowfreq_ratio"] > float(cfg.get("max_lowfreq_ratio", 0.25)):
        issues.append("low-frequency interference")
    if metrics["highfreq_ratio"] > float(cfg.get("max_highfreq_ratio", 0.35)):
        issues.append("high-frequency noise")

    return issues


def issue_text(issues: Iterable[str]) -> str:
    issues = list(issues)
    return "none" if not issues else ", ".join(issues)

