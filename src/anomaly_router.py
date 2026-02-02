import numpy as np

from src.detection_tabular import detect_tabular_anomalies
from src.detection_timeseries import detect_timeseries_anomalies
from src.detection_image import detect_image_anomaly
from src.detection_video import run_ucsd_split


def detect_anomaly(data, data_type: str, model=None):
    """
    Dispatch anomaly detection based on data_type.

    data_type: 'tabular' | 'timeseries' | 'image' | 'video'
    model: optional model for image anomaly detection (e.g., autoencoder)

    For video (UCSD baseline):
      - data can be a string path to the UCSD root folder containing UCSDped1/UCSDped2
        e.g. "data/ucsd/UCSD_Anomaly_Dataset.v1p2"
      - OR a dict:
        {"root": "...", "dataset": "UCSDped2", "split": "Test"}
    """
    data_type = data_type.lower().strip()

    if data_type == "tabular":
        return detect_tabular_anomalies(np.array(data))

    if data_type == "timeseries":
        return detect_timeseries_anomalies(np.array(data))

    if data_type == "image":
        return detect_image_anomaly(model, data)

    if data_type == "video":
        # Accept either:
        #   data = "path/to/ucsd_root"
        # or:
        #   data = {"root": "...", "dataset": "UCSDped2", "split": "Test"}
        if isinstance(data, dict):
            ucsd_root = data.get("root")
            dataset = data.get("dataset", "UCSDped1")
            split = data.get("split", "Test")
        else:
            ucsd_root = data
            dataset = "UCSDped1"
            split = "Test"

        if not ucsd_root:
            raise ValueError("For video, you must provide UCSD root path in 'data' (string or dict['root']).")

        return run_ucsd_split(
            ucsd_root=ucsd_root,
            dataset=dataset,
            split=split,
            out_dir="outputs/video_ucsd",
        )

    raise ValueError(
        f"Invalid data_type '{data_type}'. Choose: 'tabular', 'timeseries', 'image', or 'video'."
    )


if __name__ == "__main__":
    print("Running anomaly router test...")

    # Tabular quick test
    result = detect_anomaly(
        data=[1, 1, 1, 100, 1, 1],
        data_type="tabular"
    )
    print("Tabular result:", result)

    # Video test (Ped1 default)
    video_results = detect_anomaly(
        data="data/ucsd/UCSD_Anomaly_Dataset.v1p2",
        data_type="video"
    )
    print(f"Video result: wrote {len(video_results)} CSVs. First item:", video_results[0])

    # Video test (Ped2 example)
    video_results_ped2 = detect_anomaly(
        data={"root": "data/ucsd/UCSD_Anomaly_Dataset.v1p2", "dataset": "UCSDped2", "split": "Test"},
        data_type="video"
    )
    print(f"Video Ped2 result: wrote {len(video_results_ped2)} CSVs. First item:", video_results_ped2[0])
