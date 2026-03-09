from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import streamlit as st
from skimage.io import imread

# Ensure project root is importable when running from dashboard/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion
from src.fusion_evaluate import fuse_weighted_average


st.set_page_config(page_title="Multimodal Anomaly Detection Dashboard", layout="wide")

st.title("Multimodal Anomaly Detection Dashboard")
st.write(
    "Interactive dashboard for tabular, time-series, image, and video anomaly detection "
    "with explainable multimodal fusion."
)


# Sidebar controls

st.sidebar.header("Fusion Weights")
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
strong_threshold = st.sidebar.slider("Strong Threshold", 0.0, 1.0, 0.80, 0.01)
strong_threshold_video = st.sidebar.slider("Video Strong Threshold", 0.0, 1.0, 0.95, 0.01)
fused_threshold = st.sidebar.slider("Fused Threshold", 0.0, 1.0, 0.65, 0.01)
vote_k = st.sidebar.slider("Vote K", 1, 4, 2, 1)


# Inputs

st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tabular input**")
    tabular_text = st.text_input(
        "Enter tabular values (comma-separated)",
        value="1,1,1,100,1,1"
    )

    st.markdown("**Time-series input**")
    ts_text = st.text_input(
        "Enter time-series values (comma-separated)",
        value="0,0,0.1,0.2,3.5,0.1,0.0"
    )

with col2:
    st.markdown("**Image input**")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    st.markdown("**Video input**")
    default_clip = "data/ucsd/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    video_clip_path = st.text_input("UCSD clip folder path", value=default_clip)


# Helper parsers

def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]



# Run button

if st.button("Run Detection"):
    try:
        # Tabular
        tabular_data = np.asarray(parse_csv_floats(tabular_text), dtype=float)
        tab_result = detect_tabular_for_fusion(tabular_data)

        # Time-series
        ts_data = np.asarray(parse_csv_floats(ts_text), dtype=float)
        ts_result = detect_timeseries_for_fusion(ts_data, window=3)

        # Image
        if uploaded_image is not None:
            image = imread(uploaded_image)
        else:
            fallback_image = ROOT / "images" / "normal_arm.jpg"
            if fallback_image.exists():
                image = imread(str(fallback_image))
            else:
                image = np.zeros((64, 64), dtype=np.float32)

        img_result = detect_image_for_fusion(model=None, image=image)

        # Video
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

       
        # Top result
       
        st.subheader("Fusion Result")
        final_label = fused["final_label"]

        if final_label == 1:
            st.error("Final Decision: ANOMALY")
        else:
            st.success("Final Decision: NORMAL")

        st.metric("Final Fused Score", f"{fused['final_score']:.4f}")
        st.write("**Reason:**", fused["reason"])

        
        # Per-modality results
        
        st.subheader("Per-Modality Breakdown")

        modality_rows = []
        for modality, result in results.items():
            modality_rows.append({
                "modality": modality,
                "score_raw": float(result["score_raw"]),
                "score_norm": float(result["score_norm"]),
                "label": int(result["label"]),
                "method": result.get("method", ""),
            })

        st.dataframe(modality_rows, use_container_width=True)

       
        # Simple bar chart
        
        st.subheader("Normalized Scores")
        chart_data = {
            row["modality"]: row["score_norm"]
            for row in modality_rows
        }
        st.bar_chart(chart_data)

        
        # Debug / JSON view
        
        st.subheader("Raw JSON Output")
        payload = {
            "weights": weights,
            "results": results,
            "fusion": fused,
        }
        st.code(json.dumps(payload, indent=2, default=str), language="json")

    except Exception as e:
        st.exception(e)