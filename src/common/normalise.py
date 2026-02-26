import numpy as np

def normalise_percentile(x: float, p10: float, p90: float) -> float:
    """Map p10 -> 0, p90 -> 1, clamp outside."""
    if p90 <= p10:
        return 0.0
    z = (x - p10) / (p90 - p10)
    return float(np.clip(z, 0.0, 1.0))