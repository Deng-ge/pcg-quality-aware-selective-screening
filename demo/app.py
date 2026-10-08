from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install streamlit to run this optional app: pip install streamlit") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.predict import predict_record  # noqa: E402
from quality.quality_metrics import issue_text  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Heart Sound Screening Demo", layout="centered")
    st.title("Heart Sound Screening Demo")
    uploaded = st.file_uploader("Upload a wav file", type=["wav"])

    with open(PROJECT_ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if uploaded is None:
        st.info("Upload a wav file or run the CLI demo with examples/synthetic_pcg.wav.")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    result = predict_record(tmp_path, cfg)
    decision = result["decision"]

    st.metric("Prediction", decision["prediction"])
    st.metric("Confidence", f"{decision['confidence']:.2f}")
    st.metric("Quality score", f"{result['quality']['quality_score']:.2f}")
    st.write(
        {
            "normal_probability": round(result["prob_normal"], 3),
            "abnormal_probability": round(result["prob_abnormal"], 3),
            "rejected": decision["rejected"],
            "issues": issue_text(result["quality_issues"]),
            "recommendation": result["recommendation"],
        }
    )


if __name__ == "__main__":
    main()

