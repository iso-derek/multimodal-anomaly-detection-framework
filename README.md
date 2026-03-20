# Multimodal Anomaly Detection System

This repository contains my Final Year Project (FYP): a modular **multimodal anomaly detection system** that combines evidence from **tabular data, time-series data, images, and video** into one interpretable anomaly decision.

The project was developed as both a working software system and an academic investigation into how heterogeneous anomaly signals can be made comparable through **score normalisation** and combined through **decision-level fusion**.

## Overview

The system works by:

- processing each available modality with a dedicated detector
- producing a raw anomaly score for that modality
- converting scores into a shared normalised range
- combining the available modality scores through fusion
- returning a final label, fused score, and explanation

Only the modalities provided by the user are analysed in each run.

## Architecture and Workflow
![alt text](image-3.png)
![alt text](image.png)

## Current Capabilities

### Tabular
- Isolation Forest for numeric tabular anomaly detection
- float support for numeric input
- mixed tabular CSV mode for numeric + categorical data
- single-table row anomaly detection for identifying abnormal rows within one uploaded table

### Time-Series
- rolling z-score anomaly detection
- short-window spike detection for sequential anomalies

### Image
- trained autoencoder support using reconstruction error
- separate image training workflow through `train_image_autoencoder.py`
- saved model support through `models/autoencoder.keras`
- reference-bank SSIM fallback when a trained model is unavailable
- lightweight statistical fallback for robust operation
- uploaded-image analysis in the dashboard

### Video
- frame-difference based video anomaly scoring
- UCSD clip path input for video analysis
- clip-level scoring using a robust high-percentile summary

## Fusion

The fusion stage combines available modality outputs using:

- weighted average scoring
- vote-based decision logic
- strong-modality override
- support for missing modalities

The dashboard allows users to adjust **weights and thresholds** interactively. The default values are **research-informed baseline settings** used for controlled testing and interpretation.

## Dashboard

The Streamlit dashboard supports:

- numeric tabular input
- mixed tabular CSV input
- single-table tabular anomaly detection
- optional time-series input
- optional image upload
- optional video input
- adjustable fusion weights and thresholds
- per-modality result tables
- fused decision explanation
- technical JSON output

The dashboard is designed to make the system inspectable and explainable rather than acting as a black-box predictor.

## Development Summary

The project developed in three broad phases:

### Phase 1 — Research and exploratory prototyping
- literature review
- anomaly detection and multimodal fusion research
- notebook-based experimentation in Jupyter
- early testing of tabular, time-series, image, and video ideas

### Phase 2 — Structured implementation
- refactoring notebook logic into Python modules
- building modality-specific detectors
- implementing score handling and fusion
- moving the main implementation into a modular VS Code workflow

### Phase 3 — Validation and dashboard refinement
- testing the detectors and fusion pipeline
- generating result outputs
- building and refining the Streamlit dashboard
- improving flexible modality handling and tabular functionality
- polishing evaluation, interface behaviour, and final outputs

## Repository Structure

```text
Multimodial Anomaly Detection System/
├── dashboard/
│   └── app.py
├── data/
├── images/
├── models/
│   ├── autoencoder.keras
│   └── autoencoder_calibration.json
├── notebooks/
├── outputs/
├── reports/
├── src/
│   ├── common/
│   ├── anomaly_router.py
│   ├── detection_image.py
│   ├── detection_tabular.py
│   ├── detection_tabular_mixed.py
│   ├── detection_tabular_single_table.py
│   ├── detection_timeseries.py
│   ├── detection_video.py
│   ├── fusion.py
│   ├── fusion_evaluate.py
│   └── run_multimodal.py
├── tests/
├── train_image_autoencoder.py
├── README.md
└── run_tests.py
```

## Running the Project

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run the tests:

```bash
python run_tests.py
```

Train or retrain the image autoencoder:

```bash
python train_image_autoencoder.py
```

## Key Strengths

- modular architecture
- interpretable anomaly outputs
- multimodal fusion across four data types
- support for missing modalities
- layered image detection design with training + fallback support
- interactive dashboard for demonstration and analysis
- extended tabular functionality beyond basic numeric input

## Current Limitations

- some detectors are intentionally lightweight baselines
- fusion still depends on manually chosen baseline weights and thresholds
- evaluation can be expanded further with larger benchmark studies
- the image and video branches are practical, integrated implementations rather than specialist large-scale domain models

## Author

**Derek Ohimai Isokpehi**  
University of Birmingham

## Final Note

This repository presents a research-driven multimodal anomaly detection framework that combines classical machine learning, statistical methods, computer vision, and decision-level fusion in one system.

Its main contribution is not a claim of state-of-the-art performance in every individual branch, but the design and implementation of a **complete, interpretable, extensible, and defensible multimodal anomaly detection system**.
