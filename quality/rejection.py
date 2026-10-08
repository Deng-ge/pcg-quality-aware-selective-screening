from __future__ import annotations

from typing import Dict, List


def apply_rejection(
    prob_normal: float,
    prob_abnormal: float,
    quality_score: float,
    cfg: Dict | None = None,
) -> Dict:
    """Apply confidence and quality based rejection rules."""
    cfg = cfg or {}
    confidence_threshold = float(cfg.get("low_confidence_threshold", 0.60))
    quality_threshold = float(cfg.get("min_quality_score", 0.55))

    confidence = max(float(prob_normal), float(prob_abnormal))
    predicted_label = "normal" if prob_normal >= prob_abnormal else "abnormal"

    reasons: List[str] = []
    if confidence < confidence_threshold:
        reasons.append("low model confidence")
    if quality_score < quality_threshold:
        reasons.append("low signal quality")

    rejected = bool(reasons)
    return {
        "prediction": "review" if rejected else predicted_label,
        "raw_prediction": predicted_label,
        "confidence": confidence,
        "rejected": rejected,
        "reject_reasons": reasons,
    }


def recommendation(rejected: bool, quality_issues: List[str]) -> str:
    if rejected:
        if quality_issues:
            return "re-acquire the signal or send it to manual review"
        return "send to manual review"
    if quality_issues:
        return "acceptable with caution; consider re-acquisition for formal screening"
    return "acceptable for demo; for formal screening, use a stable clinical acquisition workflow"

