from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from data_preprocess.feature_logmel import compute_logmel, save_logmel_png
from data_preprocess.preprocess import bandpass_filter, center_crop_or_pad, load_wav, peak_normalize
from quality.quality_metrics import compute_quality_metrics, describe_quality_issues
from quality.rejection import apply_rejection, recommendation


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def heuristic_predict(metrics: Dict[str, float]) -> Dict[str, float]:
    """Weight-free demo backend.

    This is not a clinical classifier. It exists so the repository can be run
    without shipping private model weights.
    """
    q = metrics["quality_score"]
    high = metrics["highfreq_ratio"]
    low = metrics["lowfreq_ratio"]
    clip = metrics["clipping_ratio"]
    flat = metrics["flatline_ratio"]

    abnormal_logit = -2.0 + 2.4 * high + 1.0 * low + 5.0 * clip + 0.8 * flat + 0.8 * (1.0 - q)
    prob_abnormal = float(np.clip(_sigmoid(abnormal_logit), 0.02, 0.98))
    return {"normal": 1.0 - prob_abnormal, "abnormal": prob_abnormal}


def torch_predict(logmel: np.ndarray, weights_path: str | Path, device: str = "cpu") -> Dict[str, float]:
    import torch

    from models.resnet_attention import PCGResNetAttn

    model = PCGResNetAttn(num_classes=2).to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.tensor(logmel, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return {"normal": float(probs[0]), "abnormal": float(probs[1])}


def predict_record(
    input_path: str | Path,
    cfg: Dict,
    weights_path: str | Path | None = None,
    save_logmel_path: str | Path | None = None,
) -> Dict:
    audio_cfg = cfg.get("audio", {})
    filter_cfg = cfg.get("filter", {})
    logmel_cfg = cfg.get("logmel", {})
    quality_cfg = cfg.get("quality", {})

    target_sr = int(audio_cfg.get("sample_rate", 2000))
    clip_seconds = float(audio_cfg.get("clip_seconds", 8.0))

    raw_wav, sr = load_wav(input_path, target_sr=target_sr)
    quality = compute_quality_metrics(raw_wav, sr, quality_cfg)
    quality_issues = describe_quality_issues(quality, quality_cfg)

    wav = bandpass_filter(
        raw_wav,
        sr,
        low_hz=float(filter_cfg.get("low_hz", 20.0)),
        high_hz=float(filter_cfg.get("high_hz", 800.0)),
        order=int(filter_cfg.get("order", 4)),
    )
    wav = peak_normalize(wav)
    wav = center_crop_or_pad(wav, int(round(clip_seconds * sr)))

    logmel = compute_logmel(
        wav,
        sr=sr,
        n_mels=int(logmel_cfg.get("n_mels", 64)),
        n_fft=int(logmel_cfg.get("n_fft", 256)),
        hop_length=int(logmel_cfg.get("hop_length", 80)),
        f_min=float(logmel_cfg.get("f_min", 20.0)),
        f_max=float(logmel_cfg.get("f_max", 800.0)),
    )

    if save_logmel_path is not None:
        save_logmel_png(logmel, save_logmel_path, title="Input Log-Mel spectrogram")

    if weights_path:
        probs = torch_predict(logmel, weights_path)
        backend = "torch"
    else:
        probs = heuristic_predict(quality)
        backend = "heuristic-demo"

    decision = apply_rejection(probs["normal"], probs["abnormal"], quality["quality_score"], quality_cfg)

    return {
        "input_path": str(input_path),
        "backend": backend,
        "sample_rate": sr,
        "duration_seconds": quality["duration_seconds"],
        "prob_normal": probs["normal"],
        "prob_abnormal": probs["abnormal"],
        "quality": quality,
        "quality_issues": quality_issues,
        "decision": decision,
        "recommendation": recommendation(decision["rejected"], quality_issues),
        "logmel_shape": list(logmel.shape),
    }

