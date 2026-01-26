import numpy as np

from src.detection_tabular import detect_tabular_anomalies
from src.detection_timeseries import detect_timeseries_anomalies
from src.detection_image import detect_image_anomaly


def detect_anomaly(data, data_type: str, model=None):
    """
    Dispatch anomaly detection based on data_type.

    data_type: 'tabular' | 'timeseries' | 'image'
    model: optional model for image anomaly detection (e.g., autoencoder)
    """
    data_type = data_type.lower().strip()

    if data_type == "tabular":
        return detect_tabular_anomalies(np.array(data))

    if data_type == "timeseries":
        return detect_timeseries_anomalies(np.array(data))

    if data_type == "image":
        return detect_image_anomaly(model, data)

    raise ValueError(
        f"Invalid data_type '{data_type}'. Choose: 'tabular', 'timeseries', or 'image'."
    )

if __name__ == "__main__":
    print("Running anomaly router test...")
    result = detect_anomaly(
        data=[1, 1, 1, 100, 1, 1],
        data_type="tabular"
    )
    print(result)
