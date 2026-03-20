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
from src.detection_tabular_mixed import (
    fit_mixed_tabular_model,
    detect_mixed_tabular_for_fusion,
)
from src.detection_tabular_single_table import detect_single_table_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion
from src.fusion_evaluate import fuse_weighted_average


# Page setup
st.set_page_config(
    page_title="Multimodal Anomaly Detection Dashboard",
    layout="wide"
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

st.sidebar.header("Image Settings")
st.sidebar.caption("These control the image branch decision rule.")
image_threshold = st.sidebar.slider(
    "Image Similarity / Decision Threshold",
    0.0, 1.0, 0.35, 0.01
)


# Helpers
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
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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


# Demo defaults
DEMO_TABULAR_TEXT = "1,1\n1,1\n1,100\n1,1"
DEMO_TS_TEXT = "0,0,0.1,0.2,3.5,0.1,0.0"
DEMO_VIDEO_CLIP = "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"


# Input section
st.subheader("1. Input Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Tabular Input")

    use_tabular = st.checkbox("Use tabular modality", value=True)

    if use_tabular:
        tabular_mode = st.radio(
            "Choose tabular input type",
            ["Numeric Tabular", "Mixed Tabular CSV", "Single Table Row Anomaly Detection"],
            horizontal=True
        )

        if tabular_mode == "Numeric Tabular":
            st.caption("Demo example is pre-filled. You can edit or remove it.")
            tabular_text = st.text_area(
                "Enter numeric tabular values",
                value=DEMO_TABULAR_TEXT,
                height=120
            )
            mixed_train_csv = None
            mixed_test_csv = None
            single_table_csv = None

        elif tabular_mode == "Mixed Tabular CSV":
            st.caption("Upload a reference/training CSV and a test CSV with the same columns.")
            mixed_train_csv = st.file_uploader(
                "Upload mixed tabular training/reference CSV",
                type=["csv"],
                key="mixed_train_csv"
            )
            mixed_test_csv = st.file_uploader(
                "Upload mixed tabular test CSV",
                type=["csv"],
                key="mixed_test_csv"
            )
            single_table_csv = None
            tabular_text = ""

        else:
            st.caption("Upload one CSV table. The system will learn internal relationships and flag abnormal rows.")
            single_table_csv = st.file_uploader(
                "Upload single table CSV",
                type=["csv"],
                key="single_table_csv"
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
            value=DEMO_TS_TEXT
        )
    else:
        ts_text = ""

with col2:
    st.markdown("#### Image Input")
    use_image = st.checkbox("Use image modality", value=False)
    if use_image:
        st.caption("Upload your own image. No default image is used.")
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )
    else:
        uploaded_image = None

    st.markdown("#### Video Input")
    use_video = st.checkbox("Use video modality", value=False)
    if use_video:
        st.caption("A demo UCSD clip path is pre-filled. You can replace it with your own.")
        video_clip_path = st.text_input(
            "UCSD clip folder path",
            value=DEMO_VIDEO_CLIP
        )
    else:
        video_clip_path = ""

# Preview image only if uploaded
preview_col1, preview_col2 = st.columns([1, 2])
with preview_col1:
    if uploaded_image is not None:
        preview_img = imread(uploaded_image)
        st.image(preview_img, caption="Uploaded image", use_container_width=True)


# Run section
st.subheader("2. Run Detection")

if st.button("Run Multimodal Detection", type="primary"):
    try:
        results = {}
        tabular_input_summary = "Not used"
        mixed_train_preview = None
        mixed_test_preview = None
        single_table_preview = None
        image_source = "Not used"

        # Tabular
        if use_tabular:
            if tabular_mode == "Numeric Tabular":
                if not tabular_text.strip():
                    raise ValueError("Please enter numeric tabular data.")
                tabular_data = parse_numeric_tabular_text(tabular_text)
                tab_result = detect_tabular_for_fusion(tabular_data)
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
                tab_result = detect_mixed_tabular_for_fusion(model, test_df)
                results["tabular"] = tab_result

                tabular_input_summary = "Mixed Tabular CSV upload"
                mixed_train_preview = train_df
                mixed_test_preview = test_df

            elif tabular_mode == "Single Table Row Anomaly Detection":
                if single_table_csv is None:
                    raise ValueError("Please upload a single table CSV.")

                single_table_df = pd.read_csv(single_table_csv)
                tab_result = detect_single_table_tabular_for_fusion(single_table_df)
                results["tabular"] = tab_result

                tabular_input_summary = "Single Table Row Anomaly Detection CSV upload"
                single_table_preview = single_table_df

        # Time-series
        if use_timeseries:
            if not ts_text.strip():
                raise ValueError("Please enter time-series data or untick the time-series modality.")
            ts_data = np.asarray(parse_csv_floats(ts_text), dtype=float)
            ts_result = detect_timeseries_for_fusion(ts_data, window=3)
            results["timeseries"] = ts_result

        # Image
        if use_image:
            if uploaded_image is None:
                raise ValueError("Please upload an image or untick the image modality.")
            image = imread(uploaded_image)
            image_source = "Uploaded image"
            img_result = detect_image_for_fusion(
                model=None,
                image=image,
                threshold=image_threshold
            )
            results["image"] = img_result

        # Video
        if use_video:
            if not video_clip_path.strip():
                raise ValueError("Please provide a video clip path or untick the video modality.")
            vid_result = detect_video_clip_for_fusion(
                video_clip_path,
                threshold=0.90
            )
            results["video"] = vid_result

        if not results:
            raise ValueError("Please provide at least one modality before running detection.")

        # Fusion
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

        # Per-modality cards
        st.subheader("7. What Each Modality Means")
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
        st.subheader("8. Normalized Score Comparison")
        chart_df = pd.DataFrame({
            "Modality": [m.capitalize() for m in results.keys()],
            "Normalized Score": [float(r["score_norm"]) for r in results.values()]
        }).set_index("Modality")

        st.bar_chart(chart_df)

        # Inputs used
        st.subheader("9. Inputs Used in This Run")
        st.markdown(
            f"""
- **Tabular used:** `{use_tabular}`
- **Tabular mode:** `{tabular_mode if tabular_mode else 'Not used'}`
- **Tabular input:** `{tabular_input_summary}`
- **Time-series used:** `{use_timeseries}`
- **Time-series input:** `{ts_text if ts_text else 'Not used'}`
- **Image used:** `{use_image}`
- **Image source:** `{image_source}`
- **Image threshold:** `{image_threshold if use_image else 'Not used'}`
- **Video used:** `{use_video}`
- **Video clip path:** `{video_clip_path if video_clip_path else 'Not used'}`
"""
        )

        # Technical JSON
        with st.expander("10. Technical JSON Output"):
            payload = {
                "weights": weights,
                "thresholds": {
                    "strong_threshold": strong_threshold,
                    "strong_threshold_video": strong_threshold_video,
                    "fused_threshold": fused_threshold,
                    "vote_k": vote_k,
                    "image_threshold": image_threshold,
                },
                "results": results,
                "fusion": fused,
            }
            st.code(json.dumps(payload, indent=2, default=str), language="json")

    except Exception as e:
        st.exception(e)