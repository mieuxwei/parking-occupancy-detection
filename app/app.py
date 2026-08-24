"""Streamlit interface for single parking-space occupancy inference."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.inference import (  # noqa: E402
    ParkingOccupancyPredictor,
    decode_uploaded_image,
    resolve_checkpoint_path,
)


st.set_page_config(
    page_title="Parking Occupancy Detection",
    page_icon="🅿️",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 780px; padding-top: 2rem;}
      [data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the selected V2-A ResNet18…")
def load_predictor(checkpoint_path: str) -> ParkingOccupancyPredictor:
    return ParkingOccupancyPredictor(Path(checkpoint_path))


st.title("Parking Occupancy Detection")
st.write(
    "Upload one **cropped parking-space image** to classify it as "
    "`EMPTY` or `OCCUPIED`."
)

try:
    resolved_checkpoint = resolve_checkpoint_path(repository_root=REPOSITORY_ROOT)
    predictor = load_predictor(str(resolved_checkpoint))
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(str(error))
    st.info(
        "Set `PARKING_MODEL_PATH` to the selected V2-A ResNet18 `.pt` checkpoint, "
        "then restart Streamlit."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Parking-space crop",
    type=["jpg", "jpeg", "png", "webp"],
    help="JPEG, PNG, or WebP; maximum decoded upload size is 10 MB.",
)

if uploaded_file is None:
    st.info("Choose an image to run inference. The image stays in memory.")
else:
    try:
        image = decode_uploaded_image(uploaded_file.getvalue())
        st.image(image, caption="Uploaded parking-space crop", width="stretch")
        with st.spinner("Classifying…"):
            prediction = predictor.predict(image)
    except (ValueError, RuntimeError) as error:
        st.error(str(error))
    else:
        label = prediction["label"]
        confidence = prediction["confidence"]
        if label == "OCCUPIED":
            st.warning(f"### {label}")
        else:
            st.success(f"### {label}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)
        empty_column, occupied_column = st.columns(2)
        empty_column.metric("EMPTY probability", f"{prediction['probabilities']['EMPTY']:.1%}")
        occupied_column.metric(
            "OCCUPIED probability",
            f"{prediction['probabilities']['OCCUPIED']:.1%}",
        )

with st.expander("Scope and limitations"):
    st.markdown(
        """
        - Input must already be cropped to one parking space.
        - This demo does not locate parking spaces in a full parking-lot image.
        - Confidence is the model's softmax score, not a calibrated probability.
        - The selected V2-A model uses the locked 0.5 decision threshold.
        - Unfamiliar cameras, sites, crop boundaries, or image conditions can reduce reliability.
        """
    )
