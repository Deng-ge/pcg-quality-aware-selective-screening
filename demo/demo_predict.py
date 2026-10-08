from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.predict import predict_record  # noqa: E402
from quality.quality_metrics import issue_text  # noqa: E402


def _resolve_path(path_text: str | None, default: Path | None = None) -> Path | None:
    if path_text is None:
        return default
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the heart sound screening demo.")
    parser.add_argument("--input", default="examples/synthetic_pcg.wav", help="Input wav path.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--weights", default=None, help="Optional PyTorch checkpoint path.")
    parser.add_argument("--save-logmel", default=None, help="Optional path to save the input Log-Mel image.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(args.config)
    input_path = _resolve_path(args.input)
    weights_path = _resolve_path(args.weights) if args.weights else None
    save_logmel = _resolve_path(args.save_logmel) if args.save_logmel else None

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result = predict_record(input_path, cfg, weights_path=weights_path, save_logmel_path=save_logmel)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return

    decision = result["decision"]
    print(f"Input: {result['input_path']}")
    print(f"Backend: {result['backend']}")
    print(f"Sample rate: {result['sample_rate']} Hz")
    print(f"Duration: {result['duration_seconds']:.2f} s")
    print(f"Normal probability: {result['prob_normal']:.2f}")
    print(f"Abnormal probability: {result['prob_abnormal']:.2f}")
    print(f"Prediction: {decision['prediction']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Rejected: {'yes' if decision['rejected'] else 'no'}")
    if decision["reject_reasons"]:
        print(f"Reject reasons: {', '.join(decision['reject_reasons'])}")
    print(f"Quality score: {result['quality']['quality_score']:.2f}")
    print(f"Major quality issues: {issue_text(result['quality_issues'])}")
    print(f"Recommendation: {result['recommendation']}")
    if save_logmel:
        print(f"Saved Log-Mel image: {save_logmel}")
    print("Note: this demo is not medical advice.")


if __name__ == "__main__":
    main()

