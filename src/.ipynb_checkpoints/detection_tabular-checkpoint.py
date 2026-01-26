{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "223c8ce6-5208-4a6e-8822-0d9397bb7a8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.ensemble import IsolationForest\n",
    "from sklearn.neighbors import LocalOutlierFactor\n",
    "\n",
    "def detect_tabular_anomalies(df):\n",
    "    \"\"\"\n",
    "    Detect anomalies in tabular datasets using:\n",
    "    - Z-score\n",
    "    - Isolation Forest\n",
    "    - LOF\n",
    "    \"\"\"\n",
    "\n",
    "    # Z-score detection\n",
    "    means = df.mean()\n",
    "    stds = df.std()\n",
    "    z_scores = (df - means) / stds\n",
    "    df[\"z_anomaly\"] = (z_scores.abs() > 3).any(axis=1).astype(int)\n",
    "\n",
    "    # Isolation Forest\n",
    "    iso = IsolationForest(contamination=0.03, random_state=42)\n",
    "    df[\"if_anomaly\"] = (iso.fit_predict(df.select_dtypes(\"number\")) == -1).astype(int)\n",
    "\n",
    "    # Local Outlier Factor\n",
    "    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)\n",
    "    df[\"lof_anomaly\"] = (lof.fit_predict(df.select_dtypes(\"number\")) == -1).astype(int)\n",
    "\n",
    "    # Combine anomalies\n",
    "    df[\"combined_anomaly\"] = df[[\"z_anomaly\", \"if_anomaly\", \"lof_anomaly\"]].mean(axis=1)\n",
    "\n",
    "    return df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83e369a0-0cf8-4f57-95e8-52b0ee4bca61",
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
