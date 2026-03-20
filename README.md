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
- support for both small manual examples and uploaded CSV-based workflows
- tabular outputs summarised into fusion-ready anomaly scores

### Time-Series
- rolling z-score anomaly detection
- short-window spike detection for sequential anomalies
- support for quick manual sequence entry through the dashboard
- fusion-ready scoring using the strongest sequential anomaly signal

### Image
- trained autoencoder support using reconstruction error
- separate image training workflow through `train_image_autoencoder.py`
- automatic loading of the trained image model from `models/autoencoder.keras` when available
- automatic loading of calibration values from `models/autoencoder_calibration.json` when available
- lightweight statistical fallback for robust operation when a trained model is not available
- uploaded-image analysis in the dashboard
- single-image anomaly detection mode
- multiple-image comparison mode for comparing uploaded images against each other
- group-based image comparison using centroid-distance scoring to identify the most abnormal image within a provided set

### Video
- frame-difference based video anomaly scoring
- UCSD clip path input for video analysis
- clip-level scoring using a robust high-percentile summary
- single-video anomaly detection mode
- multiple-video comparison mode for comparing several clips against each other
- group-based video comparison using centroid-distance style summary scoring to identify the most abnormal clip within a provided set

## Fusion

The fusion stage combines available modality outputs using:

- weighted average scoring
- vote-based decision logic
- strong-modality override
- support for missing modalities

The dashboard allows users to adjust **weights and thresholds** interactively. The default values are **research-informed baseline settings** used for controlled testing and interpretation.

The fusion process is designed to remain interpretable by exposing:

- per-modality raw scores
- per-modality normalised scores
- per-modality labels
- final fused score
- plain-language explanation of the final decision

## Dashboard

The Streamlit dashboard supports:

- numeric tabular input
- mixed tabular CSV input
- single-table tabular anomaly detection
- optional time-series input
- optional image upload
- optional video input
- automatic use of the trained image autoencoder when it is present in the models folder
- automatic fallback to lightweight image scoring when no trained image model is available
- single-image anomaly detection
- multiple-image comparison
- single-video anomaly detection
- multiple-video comparison
- adjustable fusion weights and thresholds
- adjustable image decision threshold
- adjustable comparison thresholds for image and video comparison modes
- per-modality result tables
- ranked comparison outputs for image and video set analysis
- fused decision explanation
- modality-specific interpretation cards
- technical JSON output
- demo-ready default inputs for selected modes to support inspection and presentation

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
- extending the system to support image-set and video-set comparison workflows
- integrating trained-model image inference with automatic fallback behaviour
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
│   ├── detection_image_compare.py
│   ├── detection_tabular.py
│   ├── detection_tabular_mixed.py
│   ├── detection_tabular_single_table.py
│   ├── detection_timeseries.py
│   ├── detection_video.py
│   ├── detection_video_compare.py
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
- automatic trained-model loading for the image branch
- interactive dashboard for demonstration and analysis
- extended tabular functionality beyond basic numeric input
- support for both direct anomaly detection and within-set comparison workflows
- practical balance between research structure, implementation clarity, and explainability

## Current Limitations

- some detectors are intentionally lightweight baselines
- fusion still depends on manually chosen baseline weights and thresholds
- evaluation can be expanded further with larger benchmark studies
- the image and video branches are practical, integrated implementations rather than specialist large-scale domain models
- comparison modes identify relative outliers within the provided set, so their interpretation depends on the quality and representativeness of the uploaded examples

## Author

**Derek Ohimai Isokpehi**  
University of Birmingham

## Final Note

This repository presents a research-driven multimodal anomaly detection framework that combines classical machine learning, statistical methods, computer vision, and decision-level fusion in one system.

Its main contribution is not a claim of state-of-the-art performance in every individual branch, but the design and implementation of a **complete, interpretable, extensible, and defensible multimodal anomaly detection system** that supports both single-input anomaly detection and comparison-based anomaly analysis across multiple modalities.
