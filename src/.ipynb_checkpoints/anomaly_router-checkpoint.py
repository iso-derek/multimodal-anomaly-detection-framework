{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8ab689b6-6d3e-44a1-bdb8-0f62a3bc43a7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "df416548-f5b8-481d-a4f8-4fc653ffdd3d",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'src'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 2\u001b[39m\n\u001b[32m      1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnumpy\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnp\u001b[39;00m\n\u001b[32m----> \u001b[39m\u001b[32m2\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01msrc\u001b[39;00m\u001b[34;01m.\u001b[39;00m\u001b[34;01mdetection_tabular\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m detect_tabular_anomalies\n\u001b[32m      3\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01msrc\u001b[39;00m\u001b[34;01m.\u001b[39;00m\u001b[34;01mdetection_timeseries\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m detect_timeseries_anomalies\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01msrc\u001b[39;00m\u001b[34;01m.\u001b[39;00m\u001b[34;01mdetection_image\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m detect_image_anomaly\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'src'"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from src.detection_tabular import detect_tabular_anomalies\n",
    "from src.detection_timeseries import detect_timeseries_anomalies\n",
    "from src.detection_image import detect_image_anomaly\n",
    "\n",
    "def detect_anomaly(data, data_type, model=None):\n",
    "    \"\"\"\n",
    "    data_type must be one of:\n",
    "    - 'tabular'\n",
    "    - 'timeseries'\n",
    "    - 'image'\n",
    "    \"\"\"\n",
    "\n",
    "    if data_type == \"tabular\":\n",
    "        data = np.array(data)\n",
    "        return detect_tabular_anomalies(data)\n",
    "\n",
    "    elif data_type == \"timeseries\":\n",
    "        data = np.array(data)\n",
    "        return detect_timeseries_anomalies(data)\n",
    "\n",
    "    elif data_type == \"image\":\n",
    "        return detect_image_anomaly(model, data)\n",
    "\n",
    "    else:\n",
    "        raise ValueError(\n",
    "            f\"Invalid data_type '{data_type}'. \"\n",
    "            \"Choose: 'tabular', 'timeseries', or 'image'.\"\n",
    "        )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6bf7b29-97a2-49d4-87f9-d4ceb8145672",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f1918e0-a1d9-4976-9459-fd588219c7ea",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86c58574-73c8-49e6-9387-a23ea7268601",
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
