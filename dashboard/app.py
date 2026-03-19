from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
from skimage.io import imread

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion
from src.fusion_evaluate import fuse_weighted_average


# Built-in default images
DEFAULT_IMAGE_OPTIONS = [
    "normal_arm.jpg",
    "abnormal_arm.jpg",
]

IMAGE_MODEL_CANDIDATES = [
    ROOT / "models" / "image_autoencoder.keras",
    ROOT / "models" / "image_autoencoder.h5",
    ROOT / "models" / "autoencoder.keras",
    ROOT / "models" / "autoencoder.h5",
]

REFERENCE_IMAGE_DIR = ROOT / "images" / "reference_normal"


# Page setup
st.set_page_config(
    page_title="Multimodal Anomaly Detection Dashboard",
    layout="wide"
)

st.title("Multimodal Anomaly Detection Dashboard")
st.markdown(
    """
This dashboard demonstrates a **multimodal anomaly detection system**.

The system analyses four data types:

- **Tabular data** using Isolation Forest
- **Time-series data** using Rolling Z-Score
- **Image data** using an autoencoder if available, with reference-image and heuristic fallback modes
- **Video data** using frame-difference motion analysis

Each modality produces an anomaly score, and these scores are then combined using an
**explainable fusion model** to produce one final decision.
"""
)

with st.expander("How to read this dashboard"):
    st.markdown(
        """
### What this dashboard shows
- **Score Raw**: the direct anomaly score produced by the detector
- **Score Norm**: the normalized score in the range **0 to 1**
- **Label**:
  - `1` = anomaly
  - `0` = normal
- **Final Fused Score**: the combined score from all modalities
- **Reason**: the explanation for why the final decision was made

### How the fusion works
The system combines modality scores using:
1. **Weighted fusion**
2. **Strong-modality override**
3. **Threshold-based decision rules**

This means a very strong anomaly in one modality can trigger the final decision,
even if the others look normal.
"""
    )


# Helpers
def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def label_text(label: int) -> str:
    return "ANOMALY" if int(label) == 1 else "NORMAL"


def label_badge(label: int) -> str:
    return "🔴 ANOMALY" if int(label) == 1 else "🟢 NORMAL"


def explanation_for_method(method: str) -> str:
    explanations = {
        "IsolationForest_max": "This detector looks for unusual values in structured numerical data.",
        "rolling_zscore_max": "This detector checks whether a point is unusually far from recent values in the sequence.",
        "reconstruction_error": "This image score is based on autoencoder reconstruction error.",
        "reference_ssim": "This image score is based on similarity to the closest normal reference image.",
        "std_heuristic": "This image score is based on the spread of pixel values, used here as a lightweight fallback baseline.",
        "frame_diff_p95": "This video score measures strong motion changes between consecutive frames.",
    }
    return explanations.get(method, "This method produced the anomaly score for the modality.")


def build_modality_table(results: dict) -> pd.DataFrame:
    rows = []
    for modality, result in results.items():
        rows.append({
            "Modality": modality.capitalize(),
            "Method": result.get("method", ""),
            "Score Raw": round(float(result["score_raw"]), 4),
            "Score Norm": round(float(result["score_norm"]), 4),
            "Label": label_text(int(result["label"])),
        })
    return pd.DataFrame(rows)


def get_available_default_images(root: Path) -> list[str]:
    images_dir = root / "images"
    return [name for name in DEFAULT_IMAGE_OPTIONS if (images_dir / name).exists()]


def get_default_image_path(root: Path, image_name: str | None) -> Path | None:
    if not image_name:
        return None

    image_path = root / "images" / image_name
    if image_path.exists():
        return image_path
    return None


@st.cache_resource
def load_image_model():
    try:
        from tensorflow.keras.models import load_model
    except Exception as e:
        return None, f"TensorFlow/Keras not available: {e}"

    last_error = None

    for model_path in IMAGE_MODEL_CANDIDATES:
        if model_path.exists():
            try:
                model = load_model(model_path, compile=False)
                return model, f"Loaded autoencoder: {model_path.name}"
            except Exception as e:
                last_error = f"Found {model_path.name} but could not load it: {e}"

    if last_error is not None:
        return None, last_error

    return None, "No trained autoencoder model file was found."


def load_image_calibration():
    calib_path = ROOT / "models" / "autoencoder_calibration.json"
    if calib_path.exists():
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                return json.load(f), f"Loaded calibration: {calib_path.name}"
        except Exception as e:
            return None, f"Calibration file found but could not load it: {e}"
    return None, "No autoencoder calibration file was found."


image_model, image_model_status = load_image_model()
image_calib, image_calib_status = load_image_calibration()


# Sidebar settings
st.sidebar.header("Fusion Weights")
st.sidebar.caption("Higher weight means that modality has more influence on the final decision.")

video_w = st.sidebar.slider("Video Weight", 0.0, 3.0, 2.0, 0.1)
tabular_w = st.sidebar.slider("Tabular Weight", 0.0, 3.0, 1.5, 0.1)
timeseries_w = st.sidebar.slider("Time-series Weight", 0.0, 3.0, 1.0, 0.1)
image_w = st.sidebar.slider("Image Weight", 0.0, 3.0, 0.8, 0.1)

weights = {
    "video": video_w,
    "tabular": tabular_w,
    "timeseries": timeseries_w,
    "image": image_w,
}

st.sidebar.header("Fusion Thresholds")
st.sidebar.caption("These thresholds control how easily the system declares an anomaly.")

strong_threshold = st.sidebar.slider(
    "Strong Threshold (general modalities)", 0.0, 1.0, 0.80, 0.01
)
strong_threshold_video = st.sidebar.slider(
    "Strong Threshold (video only)", 0.0, 1.0, 0.95, 0.01
)
fused_threshold = st.sidebar.slider(
    "Fused Score Threshold", 0.0, 1.0, 0.65, 0.01
)
vote_k = st.sidebar.slider(
    "Minimum anomaly votes needed", 1, 4, 2, 1
)

st.sidebar.header("Image Thresholds")
st.sidebar.caption("These control the image branch decision rules.")
image_reference_threshold = st.sidebar.slider(
    "Image Reference Threshold (SSIM fallback)", 0.0, 1.0, 0.35, 0.01
)

st.sidebar.header("Image Model Status")
if image_model is not None:
    st.sidebar.success(image_model_status)
else:
    st.sidebar.warning(image_model_status)

st.sidebar.header("Image Calibration Status")
if image_calib is not None:
    st.sidebar.success(image_calib_status)
else:
    st.sidebar.warning(image_calib_status)

if REFERENCE_IMAGE_DIR.exists():
    st.sidebar.caption(f"Reference bank: {REFERENCE_IMAGE_DIR.name}")
else:
    st.sidebar.caption("Reference bank folder not found; heuristic fallback will be used if needed.")


# Input section
st.subheader("1. Input Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Tabular Input")
    st.caption("Example: `1,1,1,100,1,1` where `100` acts as a possible outlier.")
    tabular_text = st.text_input(
        "Enter tabular values (comma-separated)",
        value="1,1,1,100,1,1"
    )

    st.markdown("#### Time-Series Input")
    st.caption("Example: a sequence with one spike, which should look anomalous.")
    ts_text = st.text_input(
        "Enter time-series values (comma-separated)",
        value="0,0,0.1,0.2,3.5,0.1,0.0"
    )

with col2:
    st.markdown("#### Image Input")
    st.caption("Upload an image.")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

    available_default_images = get_available_default_images(ROOT)
    selected_default_image = None

    if available_default_images:
        selected_default_image = st.selectbox(
            "Choose built-in default image",
            available_default_images,
            index=0
        )
    else:
        st.warning("No built-in default images were found in the images folder.")

    st.markdown("#### Video Input")
    st.caption("Provide the path to a UCSD clip folder.")
    default_clip = "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    video_clip_path = st.text_input(
        "UCSD clip folder path",
        value=default_clip
    )

# Preview image
preview_col1, preview_col2 = st.columns([1, 2])
with preview_col1:
    if uploaded_image is not None:
        preview_img = imread(uploaded_image)
        st.image(preview_img, caption="Uploaded image", use_container_width=True)
    else:
        fallback_image = get_default_image_path(ROOT, selected_default_image)
        if fallback_image is not None:
            st.image(
                str(fallback_image),
                caption=f"Default image: {fallback_image.name}",
                use_container_width=True
            )


# Run section
st.subheader("2. Run Detection")

if st.button("Run Multimodal Detection", type="primary"):
    try:
        # Prepare inputs
        tabular_data = np.asarray(parse_csv_floats(tabular_text), dtype=float)
        ts_data = np.asarray(parse_csv_floats(ts_text), dtype=float)

        if uploaded_image is not None:
            image = imread(uploaded_image)
            image_source = "Uploaded image"
        else:
            fallback_image = get_default_image_path(ROOT, selected_default_image)
            if fallback_image is not None:
                image = imread(str(fallback_image))
                image_source = str(fallback_image.name)
            else:
                image = np.zeros((64, 64), dtype=np.float32)
                image_source = "Generated blank fallback image"

        # Run detectors
        tab_result = detect_tabular_for_fusion(tabular_data)
        ts_result = detect_timeseries_for_fusion(ts_data, window=3)
        img_result = detect_image_for_fusion(
            model=image_model,
            image=image,
            calib=image_calib,
            reference_dir=REFERENCE_IMAGE_DIR,
            reference_threshold=image_reference_threshold,
        )
        vid_result = detect_video_clip_for_fusion(video_clip_path, threshold=0.90)

        results = {
            "tabular": tab_result,
            "timeseries": ts_result,
            "image": img_result,
            "video": vid_result,
        }

        fused = fuse_weighted_average(
            results=results,
            weights=weights,
            strong_threshold=strong_threshold,
            strong_threshold_video=strong_threshold_video,
            fused_threshold=fused_threshold,
            vote_k=vote_k,
        )

        
        # Final result
        
        st.subheader("3. Final Fusion Result")

        final_label = int(fused["final_label"])
        final_score = float(fused["final_score"])
        final_reason = fused["reason"]

        top_left, top_right = st.columns([1, 1])

        with top_left:
            if final_label == 1:
                st.error("Final Decision: ANOMALY")
            else:
                st.success("Final Decision: NORMAL")

            st.metric("Final Fused Score", f"{final_score:.4f}")

        with top_right:
            st.markdown("#### Why did the system decide this?")
            st.info(final_reason)

       
        # Plain-English explanation
        
        st.subheader("4. Plain-English Explanation")
        st.markdown(
            f"""
- The system analysed **tabular**, **time-series**, **image**, and **video** inputs.
- Each detector produced a normalized anomaly score between **0 and 1**.
- These scores were combined using the selected fusion weights.
- The final decision was **{label_text(final_label)}**.
- The main explanation given by the fusion engine was:

> **{final_reason}**
"""
        )

        
        # Per-modality table
        
        st.subheader("5. Per-Modality Breakdown")
        modality_df = build_modality_table(results)
        st.dataframe(modality_df, use_container_width=True)

        
        # Per-modality cards
        
        st.subheader("6. What Each Modality Means")
        card_cols = st.columns(4)

        for idx, (modality, result) in enumerate(results.items()):
            with card_cols[idx]:
                st.markdown(f"### {modality.capitalize()}")
                st.write(label_badge(int(result["label"])))
                st.write(f"**Method:** {result.get('method', '')}")
                st.write(f"**Raw score:** {float(result['score_raw']):.4f}")
                st.write(f"**Normalized score:** {float(result['score_norm']):.4f}")

                if modality == "image":
                    if "best_reference" in result:
                        st.write(f"**Closest reference:** {result['best_reference']}")
                    if "best_ssim" in result:
                        st.write(f"**Best SSIM:** {float(result['best_ssim']):.4f}")
                    if "threshold" in result:
                        st.write(f"**Image threshold used:** {float(result['threshold']):.2f}")
                    if "fallback_reason" in result:
                        st.caption(f"Fallback info: {result['fallback_reason']}")

                st.caption(explanation_for_method(result.get("method", "")))

        
        # Score chart
        
        st.subheader("7. Normalized Score Comparison")
        chart_df = pd.DataFrame({
            "Modality": [m.capitalize() for m in results.keys()],
            "Normalized Score": [float(r["score_norm"]) for r in results.values()]
        }).set_index("Modality")

        st.bar_chart(chart_df)

        
        # Inputs used
        
        st.subheader("8. Inputs Used in This Run")
        st.markdown(
            f"""
- **Tabular input:** `{tabular_text}`
- **Time-series input:** `{ts_text}`
- **Image source:** `{image_source}`
- **Image model status:** `{image_model_status}`
- **Image calibration status:** `{image_calib_status}`
- **Image reference threshold:** `{image_reference_threshold:.2f}`
- **Reference image folder:** `{REFERENCE_IMAGE_DIR}`
- **Video clip path:** `{video_clip_path}`
"""
        )

        
        # Technical JSON
        
        with st.expander("9. Technical JSON Output"):
            payload = {
                "weights": weights,
                "thresholds": {
                    "strong_threshold": strong_threshold,
                    "strong_threshold_video": strong_threshold_video,
                    "fused_threshold": fused_threshold,
                    "vote_k": vote_k,
                    "image_reference_threshold": image_reference_threshold,
                },
                "image_model_status": image_model_status,
                "image_calibration_status": image_calib_status,
                "reference_image_dir": str(REFERENCE_IMAGE_DIR),
                "results": results,
                "fusion": fused,
            }
            st.code(json.dumps(payload, indent=2, default=str), language="json")

    except Exception as e:
        st.exception(e)