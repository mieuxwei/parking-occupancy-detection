"""Streamlit interface for single parking-space occupancy inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = REPOSITORY_ROOT / "app/demo_samples.json"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.inference import (  # noqa: E402
    ParkingOccupancyPredictor,
    decode_uploaded_image,
    load_locked_final_summary,
    resolve_checkpoint_path,
)


st.set_page_config(
    page_title="Parking Occupancy Detection — Cross-Domain Robustness Study",
    page_icon="🅿️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1080px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      .portfolio-kicker {color: #4f7cff; font-weight: 700; letter-spacing: .08em;}
      .portfolio-subtitle {font-size: 1.1rem; color: #667085; margin-bottom: 1.4rem;}
      .research-flow {display: flex; flex-wrap: wrap; gap: .45rem; align-items: center;}
      .research-step {padding: .42rem .7rem; border: 1px solid rgba(128,128,128,.28);
                      border-radius: 999px; font-size: .88rem;}
      .research-arrow {color: #7b8494; font-weight: 700;}
      .limitation {padding: .85rem 1rem; border-left: 4px solid #f0ad4e;
                   background: rgba(240,173,78,.09); border-radius: .35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the selected V2-A ResNet18…")
def load_predictor(checkpoint_path: str) -> ParkingOccupancyPredictor:
    return ParkingOccupancyPredictor(Path(checkpoint_path), device="cpu")


def load_demo_samples() -> list[dict[str, object]]:
    """Load the small purpose-created sample manifest."""

    samples = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    if not samples or any(sample.get("evaluation_evidence") is not False for sample in samples):
        raise ValueError("Demo sample manifest failed its non-evaluation boundary check.")
    return samples


def show_prediction(
    image,
    prediction: dict[str, object],
    image_caption: str,
    demonstration_label: str | None = None,
) -> None:
    """Render one image and the existing predictor output."""

    preview_column, result_column = st.columns([1.2, 1], gap="large")
    with preview_column:
        st.markdown("#### Image")
        st.image(image, caption=image_caption, width="stretch")
    with result_column:
        st.markdown("#### Prediction")
        label = prediction["label"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]
        if label == "OCCUPIED":
            st.warning(f"### {label}")
        else:
            st.success(f"### {label}")
        st.metric("Prediction confidence", f"{confidence:.1%}")
        st.progress(confidence)
        empty_column, occupied_column = st.columns(2)
        empty_column.metric("EMPTY score", f"{probabilities['EMPTY']:.1%}")
        occupied_column.metric("OCCUPIED score", f"{probabilities['OCCUPIED']:.1%}")
        if demonstration_label is not None:
            st.info(f"Demonstration ground truth: **{demonstration_label}**")
            st.caption("Purpose-created sample label · not an evaluation metric")


st.markdown(
    '<div class="portfolio-kicker">AI / COMPUTER VISION PORTFOLIO</div>',
    unsafe_allow_html=True,
)
st.title("Parking Occupancy Detection — Cross-Domain Robustness Study")
st.markdown(
    '<div class="portfolio-subtitle">Classify one pre-cropped parking-space image as '
    '<strong>EMPTY</strong> or <strong>OCCUPIED</strong>.</div>',
    unsafe_allow_html=True,
)

try:
    resolved_checkpoint = resolve_checkpoint_path(repository_root=REPOSITORY_ROOT)
    predictor = load_predictor(str(resolved_checkpoint))
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(f"The selected model could not be loaded: {error}")
    st.info(
        "Set `PARKING_MODEL_PATH` to the selected V2-A ResNet18 `.pt` checkpoint, "
        "then restart Streamlit."
    )
    st.stop()

try_sample_tab, upload_tab = st.tabs(["✨ Try a Sample", "⬆️ Upload Your Own"])

with try_sample_tab:
    try:
        samples = load_demo_samples()
        sample_by_id = {sample["id"]: sample for sample in samples}
        sample_ids = list(sample_by_id)
        if "demo_sample_id" not in st.session_state:
            st.session_state["demo_sample_id"] = sample_ids[0]
        another_column, _ = st.columns([1, 3])
        if another_column.button("↻ Try another sample", use_container_width=True):
            current = sample_ids.index(st.session_state["demo_sample_id"])
            st.session_state["demo_sample_id"] = sample_ids[(current + 1) % len(sample_ids)]
        selected_id = st.radio(
            "Choose a purpose-created sample",
            sample_ids,
            key="demo_sample_id",
            format_func=lambda sample_id: sample_by_id[sample_id]["title"],
            horizontal=True,
        )
        selected_sample = sample_by_id[selected_id]
        sample_path = REPOSITORY_ROOT / selected_sample["image"]
        image = decode_uploaded_image(sample_path.read_bytes())
        with st.spinner("Classifying…"):
            prediction = predictor.predict(image)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(str(error))
    else:
        show_prediction(
            image,
            prediction,
            "Purpose-created demonstration sample",
            demonstration_label=selected_sample["known_label"],
        )
        st.caption(
            "These samples are not drawn from training, validation, held-out, or "
            "fresh-final data. Source details are recorded in the repository."
        )

with upload_tab:
    st.markdown(
        "Upload a **cropped image of one parking space**. The evaluated classifier "
        "does not automatically locate spaces in a full parking-lot image."
    )
    uploaded_file = st.file_uploader(
        "Parking-space crop",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPEG, PNG, or WebP; maximum decoded upload size is 10 MB.",
    )
    if uploaded_file is None:
        st.info("Choose a cropped parking-space image. The upload stays in memory.")
    else:
        try:
            image = decode_uploaded_image(uploaded_file.getvalue())
            with st.spinner("Classifying…"):
                prediction = predictor.predict(image)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        else:
            show_prediction(image, prediction, "Uploaded parking-space crop")

st.divider()
model_column, evaluation_column = st.columns([0.9, 2.1], gap="large")
with model_column:
    st.subheader("Model")
    st.markdown("**V2-A Balanced ResNet18**")
    st.caption("224×224 · ImageNet normalization · decision threshold 0.5")

with evaluation_column:
    st.subheader("Final Evaluation")
    try:
        final_summary = load_locked_final_summary(REPOSITORY_ROOT)
    except RuntimeError:
        st.warning("Locked final-evaluation summary is temporarily unavailable.")
    else:
        accuracy_column, f1_column, recall_column = st.columns(3)
        accuracy_column.metric("Accuracy", f"{final_summary['accuracy']:.4%}")
        f1_column.metric("Occupied F1", f"{final_summary['occupied_f1']:.4%}")
        recall_column.metric(
            "UFPR04 occupied recall",
            f"{final_summary['ufpr04_occupied_recall']:.4%}",
        )
        st.caption(
            f"Locked one-time fresh-final evaluation · {final_summary['samples']:,} samples"
        )

st.subheader("Research Story")
st.markdown(
    """
    <div class="research-flow">
      <span class="research-step">Initial cross-domain model</span><span class="research-arrow">→</span>
      <span class="research-step">Domain shift discovered</span><span class="research-arrow">→</span>
      <span class="research-step">PKLot adaptation</span><span class="research-arrow">→</span>
      <span class="research-step">Error analysis</span><span class="research-arrow">→</span>
      <span class="research-step">Robustness improvement</span><span class="research-arrow">→</span>
      <span class="research-step"><strong>Final V2-A</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
st.markdown(
    """
    <div class="limitation"><strong>Scope limitation:</strong> This demo expects one
    <strong>pre-cropped parking-space image</strong>. It does not localize parking spaces or
    perform full-frame detection. Confidence is the model's softmax score and is not a
    calibrated probability.</div>
    """,
    unsafe_allow_html=True,
)
