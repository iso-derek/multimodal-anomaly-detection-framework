{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9e8672db-007f-4932-94c0-e7032ab9c4cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from aeon.anomaly_detection.series.outlier_detection import STRAY\n",
    "\n",
    "def detect_timeseries_anomalies(ts, window_size=20):\n",
    "    \"\"\"\n",
    "    STRAY time-series anomaly detection with sliding windows.\n",
    "    \"\"\"\n",
    "\n",
    "    ts = np.array(ts)\n",
    "    n = len(ts)\n",
    "\n",
    "    # Create sliding windows\n",
    "    windows = np.array([ts[i:i+window_size] for i in range(n - window_size)])\n",
    "\n",
    "    # Fit STRAY on window dataset\n",
    "    model = STRAY()\n",
    "    preds = model.fit_predict(windows)\n",
    "\n",
    "    # Align preds back to full series\n",
    "    aligned = np.zeros(n)\n",
    "    aligned[window_size:] = preds\n",
    "\n",
    "    return aligned\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb2ad09d-291c-42f7-9eaa-4a326d853563",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
