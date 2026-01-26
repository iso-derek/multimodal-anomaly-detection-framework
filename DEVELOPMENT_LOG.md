# Development Log

## Research & Prototyping
- Researched anomaly detection for tabular, time-series, and images.
- Prototyped approaches in notebooks before refactoring.

## Implementation & Refactor
- Implemented modular detectors in `src/` for tabular, time-series, and image data.
- Built a unified router to dispatch by modality.

## Environment & Tooling
- Migrated workflow from Jupyter to VS Code for stable `.py` development.
- Resolved dependency and import issues (numpy, sklearn, package structure).

## Version Control
- Connected the project to GitLab and pushed a stable baseline.
- Added `.gitignore` and began maintaining commits for incremental development.

## Next Steps
- Add video modality using frame scoring + temporal anomaly detection.
- Integrate dashboard for interactive demo and evaluation.
