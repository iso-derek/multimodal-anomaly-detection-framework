# Multimodial Ensemble Anomaly Detection System

## Overview
A modular anomaly detection framework supporting multiple data modalities.

## Current Status
- Tabular anomaly detection:  working
- Time-series anomaly detection:  working
- Image anomaly detection:  working
- Video anomaly detection:  planned (next)

## How to run
### Install dependencies
pip install numpy scikit-learn

### Run quick tests
python run_tests.py

### Run router as a module
python -m src.anomaly_router

## Structure
- src/ : core detection modules
- notebooks/ : experimentation and prototyping
- data/ : datasets
