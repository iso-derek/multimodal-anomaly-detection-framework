# Reliability-aware fusion under missing and degraded inputs

**Question:** Can input-quality weighting maintain event-level discrimination when one modality is absent or corrupted, and does any advantage survive inaccurate quality estimates?

## Alignment and partition contract

Each CSV row represents one event across tabular, time-series, image and video detector scores. Required columns are `case_id`, `group_id`, `split`, `label`, `score_<modality>`, and `quality_<modality>`. Scores and qualities lie in [0,1]; missing scores are allowed. Labels are 0 normal / 1 anomaly. Cases are unique and subject/event groups may not cross calibration, validation and test partitions.

These validations cannot prove semantic alignment: the data producer must ensure modalities refer to the same event and that detector scores were generated out of sample. Unrelated medical images and surveillance clips must not be paired and presented as an empirical multimodal dataset. The default demonstration instead generates aligned latent events and clearly labels them synthetic.

## Frozen design

- Generate 1,800 synthetic events, seed 42; 600 each for calibration, validation and testing. Each event has a shared latent abnormality and modality-specific noise.
- Learn positive modality priors from calibration bounded-score squared error: 1 / (.05 + mean squared error). These are heuristic weights, not estimated correctness probabilities. At least ten observed calibration scores are required for a modality to receive a learned prior.
- Compare equal weights, fixed learned priors, and those priors multiplied by per-event quality. Renormalise over available inputs. No strong-modality override is used in these three benchmark methods; the historical override method remains available in the interactive detector as a separate behaviour.
- Fix each method's threshold on validation negatives at a target 5% false-positive rate. Never recalibrate on corrupted test inputs. This does not guarantee 5% FPR under distribution shift.
- Evaluate clean input, four single-modality dropouts, video corruption doses .25/.5/.75/1, and full video corruption with quality left unchanged. Every method sees identical corrupted events. No test label affects weights, thresholds or scores.
- Report AUROC, average precision, threshold F1, false-positive rate and coverage. No available/reliable evidence produces abstention, not a normal label. Metrics are conditional on covered cases; coverage must always accompany them.

The synthetic corruption test supplies known corruption severity as quality information. This is a favourable assumption; the quality-unknown ablation is essential. A real application needs a separately validated quality estimator (sensor dropout, blur, missing fields, etc.) using information available at inference time. Raw scores are not calibrated event probabilities.

## Reproduce and extend

```bash
pip install -r requirements.txt
python run_tests.py
python scripts/run_research.py
python scripts/run_research.py --input aligned_scores.csv
streamlit run dashboard/research.py
```

Exports contain input cases, per-case predictions, metrics, frozen priors/thresholds, seed and input hash. The existing detector UI is `dashboard/app.py`; its quality sliders are explicitly user estimates. TensorFlow is optional for the trained image autoencoder (`requirements-autoencoder.txt`); the documented image fallback is not a trained autoencoder.

This study tests fusion mechanics and is not evidence of clinical, surveillance or operational performance. It uses one seed, does not estimate confidence intervals and does not claim academic novelty. A follow-up should use genuinely aligned independent events, multiple corruption modalities, quality-estimation error and group-resampled uncertainty. The UCSD-dependent integration test skips when the external dataset is absent.
