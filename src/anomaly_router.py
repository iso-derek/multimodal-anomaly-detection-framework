import numpy as np

from src.detection_tabular import detect_tabular_for_fusion
from src.detection_tabular_mixed import detect_mixed_tabular_for_fusion
from src.detection_tabular_single_table import detect_single_table_tabular_for_fusion
from src.detection_timeseries import detect_timeseries_for_fusion
from src.detection_image import detect_image_for_fusion
from src.detection_video import detect_video_clip_for_fusion, run_ucsd_split


def detect_anomaly(
    data,
    data_type: str,
    model=None,
    tabular_mode: str = "numeric",
    threshold: float = 0.65,
    calib: dict | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
    timeseries_window: int = 3,
    video_dataset_mode: bool = False,
):
    """
    Dispatch anomaly detection based on data type.

    data_type:
        'tabular' | 'timeseries' | 'image' | 'video'

    tabular_mode:
        'numeric' | 'mixed' | 'single_table'

    model:
        Optional model for image anomaly detection.

    threshold:
        Decision threshold for fusion-ready detectors.

    calib:
        Optional calibration dictionary with keys like p10 and p90.

    contamination:
        Isolation Forest contamination for tabular modes.

    random_state:
        Random seed for tabular modes.

    timeseries_window:
        Rolling window size for time-series detection.

    video_dataset_mode:
        If True, runs full UCSD dataset processing with run_ucsd_split.
        If False, expects one video clip/folder path and returns fusion-ready output.
    """
    data_type = data_type.lower().strip()
    tabular_mode = tabular_mode.lower().strip()

    if data_type == "tabular":
        if tabular_mode == "mixed":
            return detect_mixed_tabular_for_fusion(
                data=data,
                contamination=contamination,
                random_state=random_state,
                threshold=threshold,
                calib=calib,
            )
        elif tabular_mode == "single_table":
            return detect_single_table_tabular_for_fusion(
                data=data,
                contamination=contamination,
                random_state=random_state,
                threshold=threshold,
                calib=calib,
            )
        else:
            return detect_tabular_for_fusion(
                data=np.array(data),
                contamination=contamination,
                random_state=random_state,
                threshold=threshold,
                calib=calib,
            )

    if data_type == "timeseries":
        return detect_timeseries_for_fusion(
            series=np.array(data),
            window=timeseries_window,
            threshold=threshold,
            calib=calib,
        )

    if data_type == "image":
        return detect_image_for_fusion(
            model=model,
            image=data,
            threshold=threshold,
            calib=calib,
        )

    if data_type == "video":
        if video_dataset_mode:
            if isinstance(data, dict):
                ucsd_root = data.get("root")
                dataset = data.get("dataset", "UCSDped1")
                split = data.get("split", "Test")
            else:
                ucsd_root = data
                dataset = "UCSDped1"
                split = "Test"

            if not ucsd_root:
                raise ValueError(
                    "For dataset video mode, provide the UCSD root path in 'data' "
                    "(either as a string or dict['root'])."
                )

            return run_ucsd_split(
                ucsd_root=ucsd_root,
                dataset=dataset,
                split=split,
                out_dir="outputs/video_ucsd",
            )

        return detect_video_clip_for_fusion(
            data,
            threshold=threshold,
            calib=calib,
        )

    raise ValueError(
        f"Invalid data_type '{data_type}'. Choose: "
        f"'tabular', 'timeseries', 'image', or 'video'."
    )


if __name__ == "__main__":
    print("Running anomaly router test...")

    result_numeric = detect_anomaly(
        data=[1, 1, 1, 100, 1, 1],
        data_type="tabular",
        tabular_mode="numeric",
    )
    print("Numeric tabular result:", result_numeric)

    mixed_data = [
        {"age": 21, "department": "CS", "score": 70},
        {"age": 22, "department": "Math", "score": 72},
        {"age": 20, "department": "CS", "score": 68},
        {"age": 50, "department": "Unknown", "score": 5},
    ]
    result_mixed = detect_anomaly(
        data=mixed_data,
        data_type="tabular",
        tabular_mode="mixed",
    )
    print("Mixed tabular result:", result_mixed)

    result_single = detect_anomaly(
        data=mixed_data,
        data_type="tabular",
        tabular_mode="single_table",
    )
    print("Single-table tabular result:", result_single)

    ts = [1, 1, 1, 1, 10, 1, 1, 1]
    result_ts = detect_anomaly(
        data=ts,
        data_type="timeseries",
        timeseries_window=3,
    )
    print("Timeseries result:", result_ts)