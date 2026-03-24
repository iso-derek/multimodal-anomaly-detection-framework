from __future__ import annotations

from pathlib import Path
import sys
import json
import inspect

import numpy as np
import pandas as pd
import streamlit as st
from skimage.io import imread
from tensorflow.keras.models import load_model

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_tabular_mixed import (
    fit_mixed_tabular_model,
    detect_mixed_tabular_for_fusion,
)
from src.detection_tabular_single_table import detect_single_table_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion
from src.detection_image_compare import load_image_input, compare_image_set
from src.detection_video_compare import compare_video_set
from src.fusion_evaluate import fuse_weighted_average


# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Multimodal Anomaly Detection Dashboard",
    layout="wide",
)

st.title("Multimodal Anomaly Detection Dashboard")
st.markdown(
    """
This dashboard demonstrates a **multimodal anomaly detection system**.

The system can analyse four data types:

- **Tabular data** using Isolation Forest
- **Time-series data** using Rolling Z-Score
- **Image data** using image anomaly scoring
- **Video data** using frame-difference motion analysis

Only the modalities provided by the user are analysed in each run.
The available modality scores are then combined using an
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
- **Final Fused Score**: the combined score from all available modalities
- **Reason**: the explanation for why the final decision was made

### How the fusion works
The system combines available modality scores using:
1. **Weighted fusion**
2. **Strong-modality override**
3. **Threshold-based decision rules**

This means a very strong anomaly in one modality can trigger the final decision,
even if the others look normal.
"""
    )


# ---------------- HELPERS ----------------
def parse_numeric_tabular_text(text: str) -> np.ndarray:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("No numeric tabular data provided.")

    rows = []
    for line in lines:
        row = [float(x.strip()) for x in line.split(",") if x.strip()]
        rows.append(row)

    row_lengths = {len(r) for r in rows}
    if len(row_lengths) != 1:
        raise ValueError("All numeric tabular rows must have the same number of columns.")

    return np.asarray(rows, dtype=float)


def parse_csv_floats(text: str) -> list[float]:
    cleaned = text.replace("\n", ",").replace(" ", ",")
    return [float(x.strip()) for x in cleaned.split(",") if x.strip()]


def parse_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def label_text(label: int) -> str:
    return "ANOMALY" if int(label) == 1 else "NORMAL"


def label_badge(label: int) -> str:
    return "🔴 ANOMALY" if int(label) == 1 else "🟢 NORMAL"


def explanation_for_method(method: str) -> str:
    explanations = {
        "IsolationForest_p95": "This detector looks for unusual values in structured numerical data.",
        "IsolationForestMixed_p95": "This detector handles mixed tabular data by combining numeric scaling with categorical encoding before anomaly scoring.",
        "IsolationForestSingleTable_p95": "This detector learns relationships within one table and flags rows that look unusual compared with the rest of that dataset.",
        "rolling_zscore_max": "This detector checks whether a point is unusually far from recent values in the sequence.",
        "reconstruction_error": "This image score is based on reconstruction error from the image model.",
        "std_heuristic": "This image score is based on the spread of pixel values as a lightweight baseline.",
        "frame_diff_p95": "This video score measures strong motion changes between consecutive frames.",
        "centroid_distance_image_compare": "This compares multiple uploaded images using feature distance from the group centre.",
        "centroid_distance_video_compare": "This compares multiple video clips using summary-feature distance from the group centre.",
    }
    return explanations.get(method, "This method produced the anomaly score for the modality.")


def build_modality_table(results: dict) -> pd.DataFrame:
    rows = []
    for modality, result in results.items():
        rows.append(
            {
                "Modality": modality.capitalize(),
                "Method": result.get("method", ""),
                "Score Raw": round(float(result["score_raw"]), 4),
                "Score Norm": round(float(result["score_norm"]), 4),
                "Label": label_text(int(result["label"])),
            }
        )
    return pd.DataFrame(rows)


def image_compare_to_fusion(compare_out: dict) -> dict:
    top = compare_out.get("most_abnormal")
    if not top:
        return {
            "modality": "image",
            "score_raw": 0.0,
            "score_norm": 0.0,
            "label": 0,
            "method": "centroid_distance_image_compare",
            "meta": {"n_items": 0},
        }

    return {
        "modality": "image",
        "score_raw": float(top["score_raw"]),
        "score_norm": float(top["score_norm"]),
        "label": int(top["label"]),
        "method": "centroid_distance_image_compare",
        "meta": {
            "n_items": int(compare_out.get("n_items", 0)),
            "threshold": float(compare_out.get("threshold", 0.65)),
            "most_abnormal_name": top.get("name", ""),
            "items": compare_out.get("items", []),
            "ranked_items": compare_out.get("ranked_items", []),
        },
    }


def video_compare_to_fusion(compare_out: dict) -> dict:
    top = compare_out.get("most_abnormal")
    if not top:
        return {
            "modality": "video",
            "score_raw": 0.0,
            "score_norm": 0.0,
            "label": 0,
            "method": "centroid_distance_video_compare",
            "meta": {"n_items": 0},
        }

    return {
        "modality": "video",
        "score_raw": float(top["score_raw"]),
        "score_norm": float(top["score_norm"]),
        "label": int(top["label"]),
        "method": "centroid_distance_video_compare",
        "meta": {
            "n_items": int(compare_out.get("n_items", 0)),
            "threshold": float(compare_out.get("threshold", 0.65)),
            "most_abnormal_name": top.get("name", ""),
            "items": compare_out.get("items", []),
            "ranked_items": compare_out.get("ranked_items", []),
        },
    }


def call_with_supported_kwargs(func, *args, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(*args, **supported)


def load_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load {path.name}: {e}")
        return None


@st.cache_resource
def load_image_model():
    model_path = ROOT / "models" / "autoencoder.keras"
    if not model_path.exists():
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.warning(f"Could not load trained image model: {e}")
        return None


@st.cache_data
def load_calibration(name: str):
    candidates = [ROOT / "models" / f"{name}_calibration.json"]
    if name == "image":
        candidates.append(ROOT / "models" / "autoencoder_calibration.json")

    for path in candidates:
        calib = load_json_if_exists(path)
        if calib is not None:
            return calib
    return None


# ---------------- LOAD MODEL / CALIBRATION ----------------
image_model = load_image_model()

tabular_calib = load_calibration("tabular")
timeseries_calib = load_calibration("timeseries")
image_calib = load_calibration("image")
video_calib = load_calibration("video")


# ---------------- SIDEBAR SETTINGS ----------------
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
st.sidebar.caption("These thresholds control the final weighted fusion and vote decision.")

fused_threshold = st.sidebar.slider(
    "Fused Score Threshold", 0.0, 1.0, 0.65, 0.01
)
vote_k = st.sidebar.slider(
    "Minimum anomaly votes needed", 1, 4, 2, 1
)

st.sidebar.header("Modality Thresholds")
st.sidebar.caption("These thresholds control the anomaly sensitivity of each individual modality.")

tabular_threshold = st.sidebar.slider(
    "Tabular Threshold", 0.0, 1.0, 0.65, 0.01
)
timeseries_threshold = st.sidebar.slider(
    "Time-series Threshold", 0.0, 1.0, 0.65, 0.01
)
image_threshold = st.sidebar.slider(
    "Image Threshold", 0.0, 1.0, 0.35, 0.01
)
video_threshold = st.sidebar.slider(
    "Video Threshold", 0.0, 1.0, 0.90, 0.01
)

st.sidebar.header("Strong-Modality Override Thresholds")
st.sidebar.caption("If a modality score exceeds its strong threshold, it can trigger an override decision.")

strong_threshold_tabular = st.sidebar.slider(
    "Strong Threshold (tabular)", 0.0, 1.0, 0.80, 0.01
)
strong_threshold_timeseries = st.sidebar.slider(
    "Strong Threshold (time-series)", 0.0, 1.0, 0.80, 0.01
)
strong_threshold_image = st.sidebar.slider(
    "Strong Threshold (image)", 0.0, 1.0, 0.80, 0.01
)
strong_threshold_video = st.sidebar.slider(
    "Strong Threshold (video)", 0.0, 1.0, 0.95, 0.01
)

strong_thresholds = {
    "tabular": strong_threshold_tabular,
    "timeseries": strong_threshold_timeseries,
    "image": strong_threshold_image,
    "video": strong_threshold_video,
}

st.sidebar.header("Comparison Thresholds")
st.sidebar.caption("These thresholds are used in image-set and video-set comparison modes.")

image_compare_threshold = st.sidebar.slider(
    "Image Comparison Threshold", 0.0, 1.0, 0.65, 0.01
)
video_compare_threshold = st.sidebar.slider(
    "Video Comparison Threshold", 0.0, 1.0, 0.65, 0.01
)

if image_model is not None:
    st.sidebar.success("Loaded autoencoder: autoencoder.keras")
else:
    st.sidebar.warning("No autoencoder model loaded")

if image_calib is not None:
    st.sidebar.success("Loaded image calibration")
else:
    st.sidebar.info("No image calibration file loaded")

if tabular_calib is not None:
    st.sidebar.success("Loaded tabular calibration")
else:
    st.sidebar.info("No tabular calibration file loaded")

if timeseries_calib is not None:
    st.sidebar.success("Loaded time-series calibration")
else:
    st.sidebar.info("No time-series calibration file loaded")

if video_calib is not None:
    st.sidebar.success("Loaded video calibration")
else:
    st.sidebar.info("No video calibration file loaded")


# ---------------- DEMO DEFAULTS ----------------
DEMO_TABULAR_TEXT = "1,1\n1,1\n1,100\n1,1"
DEMO_TS_TEXT = "0,0,0.1,0.2,3.5,0.1,0.0"
DEMO_VIDEO_CLIP = "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
DEMO_MULTI_VIDEO = (
    "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001\n"
    "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test002\n"
    "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test003"
)


# ---------------- INPUT SECTION ----------------
st.subheader("1. Input Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Tabular Input")
    use_tabular = st.checkbox("Use tabular modality", value=True)

    if use_tabular:
        tabular_mode = st.radio(
            "Choose tabular input type",
            ["Numeric Tabular", "Mixed Tabular CSV", "Single Table Row Anomaly Detection"],
            horizontal=True,
        )

        if tabular_mode == "Numeric Tabular":
            st.caption("Demo example is pre-filled. You can edit or remove it.")
            tabular_text = st.text_area(
                "Enter numeric tabular values",
                value=DEMO_TABULAR_TEXT,
                height=120,
            )
            mixed_train_csv = None
            mixed_test_csv = None
            single_table_csv = None

        elif tabular_mode == "Mixed Tabular CSV":
            st.caption("Upload a reference/training CSV and a test CSV with the same columns.")
            mixed_train_csv = st.file_uploader(
                "Upload mixed tabular training/reference CSV",
                type=["csv"],
                key="mixed_train_csv",
            )
            mixed_test_csv = st.file_uploader(
                "Upload mixed tabular test CSV",
                type=["csv"],
                key="mixed_test_csv",
            )
            single_table_csv = None
            tabular_text = ""

        else:
            st.caption("Upload one CSV table. The system will learn internal relationships and flag abnormal rows.")
            single_table_csv = st.file_uploader(
                "Upload single table CSV",
                type=["csv"],
                key="single_table_csv",
            )
            mixed_train_csv = None
            mixed_test_csv = None
            tabular_text = ""
    else:
        tabular_mode = None
        tabular_text = ""
        mixed_train_csv = None
        mixed_test_csv = None
        single_table_csv = None

    st.markdown("#### Time-Series Input")
    use_timeseries = st.checkbox("Use time-series modality", value=True)
    if use_timeseries:
        st.caption("Demo example is pre-filled. You can edit or remove it.")
        ts_text = st.text_input(
            "Enter time-series values (comma-separated)",
            value=DEMO_TS_TEXT,
        )
    else:
        ts_text = ""

with col2:
    st.markdown("#### Image Input")
    use_image = st.checkbox("Use image modality", value=False)

    if use_image:
        if image_model is not None:
            st.caption("Trained image model loaded: models/autoencoder.keras")
        else:
            st.caption("No trained image model loaded. Image branch will use fallback scoring.")

        image_mode = st.radio(
            "Image analysis mode",
            ["Single Image", "Compare Multiple Images"],
            horizontal=True,
        )

        if image_mode == "Single Image":
            st.caption("Upload your own image. No default image is used.")
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                key="single_image_upload",
            )
            uploaded_images = []
        else:
            st.caption("Upload multiple images to compare them against each other.")
            uploaded_images = st.file_uploader(
                "Upload multiple images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="multi_image_upload",
            )
            uploaded_image = None
    else:
        image_mode = None
        uploaded_image = None
        uploaded_images = []

    st.markdown("#### Video Input")
    use_video = st.checkbox("Use video modality", value=False)

    if use_video:
        video_mode = st.radio(
            "Video analysis mode",
            ["Single Video", "Compare Multiple Videos"],
            horizontal=True,
        )

        if video_mode == "Single Video":
            st.caption("A demo UCSD clip path is pre-filled. You can replace it.")
            video_clip_path = st.text_input(
                "UCSD clip folder path",
                value=DEMO_VIDEO_CLIP,
            )
            multi_video_paths_text = ""
        else:
            st.caption("Demo clip paths are pre-filled. You can replace or remove them.")
            multi_video_paths_text = st.text_area(
                "Video clip folder paths (one per line)",
                value=DEMO_MULTI_VIDEO,
                height=120,
            )
            video_clip_path = ""
    else:
        video_mode = None
        video_clip_path = ""
        multi_video_paths_text = ""


# ---------------- PREVIEW ----------------
preview_col1, preview_col2 = st.columns([1, 2])
with preview_col1:
    if use_image:
        if image_mode == "Single Image" and uploaded_image is not None:
            uploaded_image.seek(0)
            preview_img = imread(uploaded_image)
            st.image(preview_img, caption="Uploaded image", use_container_width=True)

        elif image_mode == "Compare Multiple Images" and uploaded_images:
            st.caption(f"{len(uploaded_images)} images uploaded for comparison")
            st.write("Uploaded files:")
            for f in uploaded_images:
                st.write(f"- {f.name}")


# ---------------- RUN SECTION ----------------
st.subheader("2. Run Detection")

if st.button("Run Multimodal Detection", type="primary"):
    try:
        results = {}
        tabular_input_summary = "Not used"
        mixed_train_preview = None
        mixed_test_preview = None
        single_table_preview = None
        image_source = "Not used"
        video_source = "Not used"
        image_compare_out = None
        video_compare_out = None

        # Tabular
        if use_tabular:
            if tabular_mode == "Numeric Tabular":
                if not tabular_text.strip():
                    raise ValueError("Please enter numeric tabular data.")
                tabular_data = parse_numeric_tabular_text(tabular_text)
                tab_result = call_with_supported_kwargs(
                    detect_tabular_for_fusion,
                    tabular_data,
                    threshold=tabular_threshold,
                    calib=tabular_calib,
                )
                results["tabular"] = tab_result
                tabular_input_summary = tabular_text

            elif tabular_mode == "Mixed Tabular CSV":
                if mixed_train_csv is None:
                    raise ValueError("Please upload a mixed tabular training/reference CSV.")
                if mixed_test_csv is None:
                    raise ValueError("Please upload a mixed tabular test CSV.")

                train_df = pd.read_csv(mixed_train_csv)
                test_df = pd.read_csv(mixed_test_csv)

                model = fit_mixed_tabular_model(train_df)
                tab_result = call_with_supported_kwargs(
                    detect_mixed_tabular_for_fusion,
                    model,
                    test_df,
                    threshold=tabular_threshold,
                    calib=tabular_calib,
                )
                results["tabular"] = tab_result

                tabular_input_summary = "Mixed Tabular CSV upload"
                mixed_train_preview = train_df
                mixed_test_preview = test_df

            elif tabular_mode == "Single Table Row Anomaly Detection":
                if single_table_csv is None:
                    raise ValueError("Please upload a single table CSV.")

                single_table_df = pd.read_csv(single_table_csv)
                tab_result = call_with_supported_kwargs(
                    detect_single_table_tabular_for_fusion,
                    single_table_df,
                    threshold=tabular_threshold,
                    calib=tabular_calib,
                )
                results["tabular"] = tab_result

                tabular_input_summary = "Single Table Row Anomaly Detection CSV upload"
                single_table_preview = single_table_df

        # Time-series
        if use_timeseries:
            if not ts_text.strip():
                raise ValueError("Please enter time-series data or untick the time-series modality.")
            ts_data = np.asarray(parse_csv_floats(ts_text), dtype=float)
            ts_result = call_with_supported_kwargs(
                detect_timeseries_for_fusion,
                ts_data,
                window=3,
                threshold=timeseries_threshold,
                calib=timeseries_calib,
            )
            results["timeseries"] = ts_result

        # Image
        if use_image:
            if image_mode == "Single Image":
                if uploaded_image is None:
                    raise ValueError("Please upload an image or change the image mode.")
                uploaded_image.seek(0)
                image = imread(uploaded_image)
                image_source = "Uploaded image"
                img_result = call_with_supported_kwargs(
                    detect_image_for_fusion,
                    model=image_model,
                    image=image,
                    threshold=image_threshold,
                    calib=image_calib,
                )
                results["image"] = img_result

            elif image_mode == "Compare Multiple Images":
                if len(uploaded_images) < 2:
                    raise ValueError("Please upload at least two images for image comparison.")
                image_arrays = [load_image_input(f) for f in uploaded_images]
                image_names = [f.name for f in uploaded_images]

                image_compare_out = compare_image_set(
                    images=image_arrays,
                    names=image_names,
                    threshold=image_compare_threshold,
                )
                img_result = image_compare_to_fusion(image_compare_out)
                results["image"] = img_result
                image_source = f"{len(uploaded_images)} uploaded images compared"

        # Video
        if use_video:
            if video_mode == "Single Video":
                if not video_clip_path.strip():
                    raise ValueError("Please provide a video clip path or change the video mode.")
                vid_result = call_with_supported_kwargs(
                    detect_video_clip_for_fusion,
                    video_clip_path,
                    threshold=video_threshold,
                    calib=video_calib,
                )
                results["video"] = vid_result
                video_source = video_clip_path

            elif video_mode == "Compare Multiple Videos":
                clip_paths = parse_lines(multi_video_paths_text)
                if len(clip_paths) < 2:
                    raise ValueError("Please provide at least two clip paths for video comparison.")

                video_compare_out = compare_video_set(
                    clip_paths=clip_paths,
                    names=[Path(p).name for p in clip_paths],
                    threshold=video_compare_threshold,
                )
                vid_result = video_compare_to_fusion(video_compare_out)
                results["video"] = vid_result
                video_source = f"{len(clip_paths)} clip paths compared"

        if not results:
            raise ValueError("Please provide at least one modality before running detection.")

        # Fusion
        fused = call_with_supported_kwargs(
            fuse_weighted_average,
            results=results,
            weights=weights,
            fused_threshold=fused_threshold,
            vote_k=vote_k,
            strong_thresholds=strong_thresholds,
            strong_threshold=max(
                strong_threshold_tabular,
                strong_threshold_timeseries,
                strong_threshold_image,
            ),
            strong_threshold_video=strong_threshold_video,
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
- The system analysed **{', '.join(results.keys())}**.
- Only the modalities provided by the user were run.
- Each available detector produced a normalized anomaly score between **0 and 1**.
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

        # Mixed CSV preview
        if tabular_mode == "Mixed Tabular CSV" and mixed_train_preview is not None and mixed_test_preview is not None:
            st.subheader("6. Mixed Tabular CSV Preview")
            left_csv, right_csv = st.columns(2)

            with left_csv:
                st.markdown("**Training / Reference CSV**")
                st.dataframe(mixed_train_preview.head(), use_container_width=True)

            with right_csv:
                st.markdown("**Test CSV**")
                st.dataframe(mixed_test_preview.head(), use_container_width=True)

        # Single table preview
        if tabular_mode == "Single Table Row Anomaly Detection" and single_table_preview is not None:
            st.subheader("6A. Single Table CSV Preview")
            st.dataframe(single_table_preview.head(), use_container_width=True)

            anomalous_rows = results.get("tabular", {}).get("anomalous_rows", [])
            if anomalous_rows:
                st.subheader("6B. Abnormal Rows Found")
                st.dataframe(pd.DataFrame(anomalous_rows), use_container_width=True)
            else:
                st.info("No abnormal rows were flagged in the uploaded table.")

        # Image comparison results
        if image_compare_out is not None:
            st.subheader("7. Image Comparison Results")
            st.write("Images are ranked using centroid distance from the group centre, with SSIM shown as supporting similarity evidence.")
            st.dataframe(pd.DataFrame(image_compare_out["ranked_items"]), use_container_width=True)

        # Video comparison results
        if video_compare_out is not None:
            st.subheader("8. Video Comparison Results")
            st.write("Video clips are ranked by distance from the group centre using compact motion-based detector summaries.")
            st.dataframe(pd.DataFrame(video_compare_out["ranked_items"]), use_container_width=True)

        # Per-modality cards
        st.subheader("9. What Each Modality Means")
        card_cols = st.columns(len(results))

        for idx, (modality, result) in enumerate(results.items()):
            with card_cols[idx]:
                st.markdown(f"### {modality.capitalize()}")
                st.write(label_badge(int(result["label"])))
                st.write(f"**Method:** {result.get('method', '')}")
                st.write(f"**Raw score:** {float(result['score_raw']):.4f}")
                st.write(f"**Normalized score:** {float(result['score_norm']):.4f}")
                st.caption(explanation_for_method(result.get("method", "")))

        # Score chart
        st.subheader("10. Normalized Score Comparison")
        chart_df = pd.DataFrame(
            {
                "Modality": [m.capitalize() for m in results.keys()],
                "Normalized Score": [float(r["score_norm"]) for r in results.values()],
            }
        ).set_index("Modality")

        st.bar_chart(chart_df)

        # Inputs used
        st.subheader("11. Inputs Used in This Run")
        st.markdown(
            f"""
- **Tabular used:** `{use_tabular}`
- **Tabular mode:** `{tabular_mode if tabular_mode else 'Not used'}`
- **Tabular input:** `{tabular_input_summary}`
- **Tabular threshold:** `{tabular_threshold if use_tabular else 'Not used'}`
- **Tabular strong threshold:** `{strong_threshold_tabular if use_tabular else 'Not used'}`
- **Tabular calibration loaded:** `{tabular_calib is not None}`

- **Time-series used:** `{use_timeseries}`
- **Time-series input:** `{ts_text if ts_text else 'Not used'}`
- **Time-series threshold:** `{timeseries_threshold if use_timeseries else 'Not used'}`
- **Time-series strong threshold:** `{strong_threshold_timeseries if use_timeseries else 'Not used'}`
- **Time-series calibration loaded:** `{timeseries_calib is not None}`

- **Image used:** `{use_image}`
- **Image mode:** `{image_mode if image_mode else 'Not used'}`
- **Image source:** `{image_source}`
- **Image threshold:** `{image_threshold if use_image else 'Not used'}`
- **Image strong threshold:** `{strong_threshold_image if use_image else 'Not used'}`
- **Image comparison threshold:** `{image_compare_threshold if use_image and image_mode == 'Compare Multiple Images' else 'Not used'}`
- **Image calibration loaded:** `{image_calib is not None}`

- **Video used:** `{use_video}`
- **Video mode:** `{video_mode if video_mode else 'Not used'}`
- **Video source:** `{video_source}`
- **Video threshold:** `{video_threshold if use_video and video_mode == 'Single Video' else 'Not used'}`
- **Video strong threshold:** `{strong_threshold_video if use_video else 'Not used'}`
- **Video comparison threshold:** `{video_compare_threshold if use_video and video_mode == 'Compare Multiple Videos' else 'Not used'}`
- **Video calibration loaded:** `{video_calib is not None}`
"""
        )

        # Technical JSON
        with st.expander("12. Technical JSON Output"):
            payload = {
                "weights": weights,
                "thresholds": {
                    "fused_threshold": fused_threshold,
                    "vote_k": vote_k,
                    "tabular_threshold": tabular_threshold,
                    "timeseries_threshold": timeseries_threshold,
                    "image_threshold": image_threshold,
                    "video_threshold": video_threshold,
                    "strong_thresholds": strong_thresholds,
                    "image_compare_threshold": image_compare_threshold,
                    "video_compare_threshold": video_compare_threshold,
                },
                "calibration_loaded": {
                    "tabular": tabular_calib is not None,
                    "timeseries": timeseries_calib is not None,
                    "image": image_calib is not None,
                    "video": video_calib is not None,
                },
                "results": results,
                "fusion": fused,
                "image_compare": image_compare_out,
                "video_compare": video_compare_out,
                "image_model_loaded": image_model is not None,
            }
            st.code(json.dumps(payload, indent=2, default=str), language="json")

    except Exception as e:
        st.exception(e)