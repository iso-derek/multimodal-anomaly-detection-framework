# Multimodal Anomaly Detection System

This repository contains my Final Year Project (FYP) at the University of Birmingham. The project presents a modular **multimodal anomaly detection framework** that combines evidence from **tabular data, time-series data, images, and video** through calibrated scoring and interpretable decision-level fusion.

The motivation behind the project is that anomalies in real-world systems do not always appear in just one form. Abnormal behaviour may be reflected in unusual structured values, irregular temporal patterns, visual differences in images, or unexpected motion in video. This project brings those different signals together in one unified framework.

The final system is supported by both a modular Python implementation and an interactive Streamlit dashboard, allowing the framework to be demonstrated, analysed, and explained more effectively.

---

## Project Overview

Traditional anomaly detection approaches are often designed for a single data type. In contrast, this project explores how anomaly detection can be improved by combining multiple modalities within one pipeline.

The system works by:

- processing each modality with a dedicated detector,
- producing a raw anomaly score for each modality,
- calibrating those scores into a comparable range,
- and combining them through a fusion stage to generate a final anomaly decision.

The final output includes:

- raw anomaly scores,
- normalised anomaly scores,
- per-modality labels,
- a fused final score,
- and an explanation of why the sample was classified as normal or anomalous.

The project was designed not only as a working software artefact, but also as an academic investigation into how heterogeneous anomaly evidence can be combined in an interpretable and extensible way.

---

## Aim and Objectives

The overall aim of the project is to develop a practical, interpretable, and extensible multimodal anomaly detection framework suitable for experimentation, evaluation, and demonstration in a Final Year Project setting.

### Objectives

1. Build separate anomaly detectors for:
   - tabular data,
   - time-series data,
   - image data,
   - video data.
2. Transform raw detector outputs into a common calibrated score range.
3. Design a fusion mechanism that combines evidence across modalities.
4. Produce interpretable outputs that can be inspected and explained.
5. Demonstrate the framework on both normal and abnormal examples.
6. Provide a usable front-end dashboard for running and visualising the system.

---

## Core Features

- Support for **four different data modalities**
- Dedicated detector for each modality
- Score normalisation across heterogeneous anomaly outputs
- Decision-level fusion using weighted scoring, vote logic, and strong-modality override
- Interpretable outputs showing how each modality contributed to the final result
- Interactive dashboard for running multimodal anomaly detection experiments
- Adjustable fusion weights and thresholds through the user interface
- Per-modality result tables, score visualisation, and raw JSON output for transparency
- Modular repository structure that supports extension and experimentation

---

## Detection Pipeline

### 1. Tabular Anomaly Detection

The tabular module analyses structured numeric features and identifies unusual observations using a classical unsupervised anomaly detection method.

**Current implementation**
- Isolation Forest
- Percentile-based thresholding (p95)

**Rationale**
This method was selected as a strong baseline because it is well suited to unsupervised anomaly detection, performs effectively when anomalies are rare, and remains computationally lightweight and interpretable.

---

### 2. Time-Series Anomaly Detection

The time-series module detects abnormal behaviour in sequential data by identifying local deviations over time.

**Current implementation**
- Rolling z-score
- Window size = 3

**Rationale**
This approach was deliberately chosen as a transparent and computationally efficient baseline. It is effective for identifying short-term spikes, unusual shifts, and local temporal irregularities without requiring a complex learned model.

---

### 3. Image Anomaly Detection

The image module compares visual inputs against expected normal behaviour and assigns an anomaly score based on image characteristics.

**Current implementation**
- 3-channel image support
- Standard deviation-based anomaly heuristic
- Normal vs abnormal X-ray comparison workflow

**Rationale**
The image component was implemented as a lightweight visual anomaly baseline that could be integrated into the multimodal framework without requiring heavy training infrastructure. This made it suitable for prototype development while still demonstrating image-based anomaly handling.

---

### 4. Video Anomaly Detection

The video module analyses motion behaviour across frames and detects unusual activity patterns.

**Current implementation**
- UCSD-style frame difference baseline
- Percentile-based thresholding (p95)
- Anomaly threshold = 0.90
- Strong-modality override threshold = 0.95

**Rationale**
Motion irregularity is a useful signal for detecting anomalies in video. Frame differencing provides a clear and computationally feasible baseline for anomaly scoring and integrates well into the overall fusion framework.

---

## Fusion Strategy

After each modality produces a result, the system combines them into one final anomaly decision.

### Fusion design

The fusion stage uses:

- a **weighted average** of normalised modality scores,
- a **vote gate** to capture agreement across modalities,
- and a **strong-modality override** to allow a highly confident detector to trigger an anomaly decision on its own.

### Why this matters

This fusion design improves robustness because:

- one weak detector is less likely to dominate the overall result,
- strongly abnormal evidence can still be acted upon even if other modalities are less confident,
- and the final decision remains interpretable through the individual modality scores and labels.

This fusion layer is one of the main contributions of the project, as it allows heterogeneous detectors to be integrated into a single explainable anomaly decision process.

---

## Example Output

A typical fused result looks like this:

```json
{
  "final_label": "anomaly",
  "final_score": 0.93,
  "reason": "video strong override triggered",
  "modalities": {
    "tabular": {"score_raw": 0.61, "score_norm": 0.70, "label": "normal"},
    "timeseries": {"score_raw": 2.85, "score_norm": 0.76, "label": "anomaly"},
    "image": {"score_raw": 0.49, "score_norm": 0.68, "label": "normal"},
    "video": {"score_raw": 0.97, "score_norm": 0.98, "label": "anomaly"}
  }
}
```

Example saved outputs from the current system include:

- `fusion_summary_normal_arm.json`
- `fusion_summary_abnormal_arm.json`
- `fusion_summary.json`

---

## System Architecture

At a high level, the project follows this flow:

```text
src/run_multimodal.py
    ↓
src/anomaly_router.py
    ↓
src/detection_tabular.py
src/detection_timeseries.py
src/detection_image.py
src/detection_video.py
    ↓
src/fusion.py
    ↓
outputs/final fused output (score + label + reason)
```

### Component summary

- **`src/run_multimodal.py`** is the main entry point for the multimodal pipeline.
- **`src/anomaly_router.py`** coordinates the detection workflow and manages detector outputs.
- **`src/detection_*`** files contain modality-specific anomaly scoring logic.
- **`src/fusion.py`** combines the detector outputs into one final decision.
- **`outputs/`** stores JSON summaries and generated artefacts used for evaluation and reporting.

---

## Notebook-Based Prototyping and Early Development

Before the project was refactored into a modular Python pipeline, the early stages of development were carried out in **Jupyter notebooks**. These notebooks were used as an experimental workspace to explore datasets, test baseline methods, visualise intermediate results, and compare anomaly scoring behaviour across modalities.

Each modality was first explored independently:

- tabular anomaly detection was tested on structured feature data,
- time-series behaviour was analysed using rolling statistical methods,
- image anomaly ideas were explored through visual comparison and heuristics,
- and video anomaly experiments were used to study frame-difference behaviour before integration into the final system.

This notebook-first approach was important because it made it easier to validate ideas quickly, debug preprocessing steps, and refine thresholds before transferring the logic into reusable Python modules.

In this sense, the notebooks represent the experimentation and research phase of the project, while the `src/` pipeline represents the final system implementation.

---


## Development Timeline

The project was developed in three broad phases.

### Phase 1 — Research and exploratory prototyping (October–December)

- literature review
- method selection
- anomaly detection theory
- Jupyter notebook experiments
- testing initial ideas for tabular, time-series, and image modalities

### Phase 2 — Structured implementation (January–February)

- refactoring notebook work into Python modules
- building modality-specific detectors
- implementing fusion logic
- integrating the multimodal pipeline

### Phase 3 — Validation and system completion (late February–March)

- testing detectors and fusion
- evaluating image and video inputs
- generating results
- building the Streamlit dashboard
- finalising the report

---

## Dashboard

In addition to the Python pipeline, the project includes an interactive dashboard built with **Streamlit**. The dashboard was developed to make the system easier to demonstrate, inspect, and evaluate during the final project presentation.

The dashboard allows the user to:

- enter tabular input values,
- enter time-series values,
- upload an image,
- provide a UCSD video clip path,
- adjust modality fusion weights,
- adjust fusion thresholds,
- run the multimodal detection pipeline from one interface,
- inspect the fused anomaly decision,
- view per-modality scores and labels,
- compare normalised modality scores visually,
- and inspect the raw JSON output generated by the system.

This strengthens the project by showing that the framework is not only implemented at code level, but also usable through a clear interactive interface for testing and demonstration.

---

## Repository Structure

The simplified project structure is shown below. Temporary development folders such as `__pycache__` and `.ipynb_checkpoints` are omitted here for clarity.

```text
FYP FYP/
└── Multimodial Anomaly Detection System/
    ├── .github/
    ├── .vscode/
    ├── launch.json
    ├── dashboard/
    │   └── app.py
    ├── data/
    │   └── ucsd/
    │       └── UCSD_Anomaly_Dataset.v1p2/
    │           ├── UCSDped1/
    │           ├── UCSDped2/
    │           └── README.txt
    ├── images/
    │   ├── abnormal_arm.jpg
    │   └── normal_arm.jpg
    ├── notebooks/
    │   ├── 01_tabular_*.ipynb
    │   ├── 02_time_series_*.ipynb
    │   └── 03_image_anomaly.ipynb
    ├── outputs/
    │   ├── video_ucsd/
    │   ├── fusion_summary.json
    │   ├── fusion_summary_abnormal_*.json
    │   └── fusion_summary_normal_*.json
    ├── reports/
    ├── src/
    │   ├── common/
    │   │   ├── __init__.py
    │   │   ├── normalise.py
    │   │   └── result_types.py
    │   ├── __init__.py
    │   ├── anomaly_router.py
    │   ├── detection_image.py
    │   ├── detection_tabular.py
    │   ├── detection_timeseries.py
    │   ├── detection_video.py
    │   ├── fusion.py
    │   ├── fusion_evaluate.py
    │   ├── plot_video_scores.py
    │   ├── plot_video_segments.py
    │   ├── run_multimodal.py
    │   └── video_postprocess.py
    ├── tests/
    │   ├── test_fusion.py
    │   ├── test_image.py
    │   ├── test_tabular.py
    │   ├── test_timeseries.py
    │   └── test_video.py
    ├── .gitignore
    ├── DEVELOPMENT_LOG.md
    ├── python
    ├── README.md
    ├── run_tests.py
    ├── TODO.md
    └── REPORT/
        ├── Excellent.odt
        └── FYP.odt
```

### Structure notes

- **`dashboard/`** contains the Streamlit-based interactive interface for running the multimodal anomaly detection system, adjusting fusion parameters, uploading inputs, and inspecting results visually.
- **`data/`** stores the UCSD anomaly dataset used by the video module.
- **`images/`** contains the normal and abnormal arm X-ray examples used by the image module.
- **`notebooks/`** contains the early Jupyter notebook experiments used during prototyping.
- **`outputs/`** stores fused JSON summaries and related result artefacts.
- **`src/`** contains the core implementation of the multimodal detection and fusion framework.
- **`tests/`** contains module-level tests for fusion, tabular, time-series, image, and video components.
- **`DEVELOPMENT_LOG.md`** and **`TODO.md`** document project progress and next steps.
- **`REPORT/`** contains dissertation working documents.

---

## Technologies Used

The project is implemented primarily in Python and combines methods from machine learning, anomaly detection, and computer vision.

Libraries and tools used across the project include:

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- OpenCV
- scikit-image
- Matplotlib
- JSON and standard Python utilities
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd "Multimodial Anomaly Detection System"
```

Install the required dependencies using your preferred Python environment.

---

## Running the Project

Run the main multimodal pipeline:

```bash
python src/run_multimodal.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the test suite:

```bash
python run_tests.py
```

---

## Input Data

The framework is designed to work with four broad input types.

### Tabular
Structured data such as CSV files containing numerical or engineered features.

### Time-Series
Sequential observations such as sensor values, logs, or measurements over time.

### Images
Still images, including the X-ray examples used in this project.

### Video
Short clips or frame sequences, including UCSD-style anomaly examples.

---

## Outputs

The project generates outputs that can be used for analysis, demonstration, and dissertation evidence.

These may include:

- per-modality anomaly scores,
- normalised scores,
- modality-level labels,
- fused final decisions,
- reasoning text explaining the decision,
- JSON summaries,
- score visualisations,
- and dashboard-based output views.

---

## Research Contribution

The main contribution of this project is a practical framework for **heterogeneous anomaly detection**. Rather than treating each data type separately, it demonstrates how evidence from different modalities can be brought together into a single interpretable decision process.

The contribution is not limited to anomaly detection alone. It also includes:

- calibration of scores across different detectors,
- decision-level fusion of heterogeneous evidence,
- interpretability through modality-specific outputs,
- modular implementation for reuse and extension,
- and an interactive dashboard for demonstration and analysis.

This makes the framework relevant to settings where one data source alone does not provide the full picture.

---

## Strengths of the Project

Some of the key strengths of the project are:

- clear modular system architecture,
- support for multiple modalities in one framework,
- interpretable decision-making through per-modality outputs,
- a usable interactive dashboard for demonstration,
- computationally practical baseline methods,
- and a strong foundation for future extension and evaluation.

---

## Current Limitations

The current system delivers a working multimodal anomaly detection framework with an interactive dashboard, but there are still areas where the implementation could be extended further.

- Some modality detectors are deliberately implemented as interpretable baseline methods rather than more complex learned models.
- Fusion currently depends on manually configurable weights and thresholds rather than data-driven optimisation.
- Performance is influenced by the quality and representativeness of the available datasets for each modality.
- The dashboard is functional and suitable for demonstration, but it can still be refined into a more polished and feature-complete interface.
- Evaluation can be expanded further using larger benchmarks, more systematic ablation studies, and additional quantitative performance metrics.

---

## Future Improvements

There are several realistic directions for extending the project further:

- integrate more advanced learned models for selected modalities,
- explore optical flow or deep spatiotemporal approaches for video anomaly detection,
- investigate transformer-based or deep sequence models for time-series anomaly detection,
- develop more adaptive or learned fusion strategies,
- improve handling of missing or incomplete modalities,
- extend the dashboard with richer visual analytics and upload workflows,
- and broaden benchmarking against stronger unimodal and multimodal baselines.

---

## Possible Use Cases

Although the project was developed as an academic prototype, the overall framework could be adapted to areas such as:

- healthcare monitoring,
- industrial fault detection,
- surveillance and activity monitoring,
- sensor anomaly detection,
- quality inspection,
- and intelligent monitoring systems.

---


## Author

**Derek Ohimai Isokpehi**  
BSc Computer Science  
University of Birmingham

---

## License

This project is intended for academic and research purposes only.

---

## Final Note

This repository represents a research-driven multimodal anomaly detection prototype that combines classical machine learning, statistical methods, computer vision, and decision-level fusion in one system.

It was developed both as a functional implementation and as an academic investigation into how heterogeneous anomaly evidence can be combined into a single explainable anomaly decision.
