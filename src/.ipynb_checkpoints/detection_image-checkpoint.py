{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9a13a10c-1136-41c3-a697-fac06aa446bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "import torch\n",
    "from skimage.metrics import structural_similarity as ssim\n",
    "\n",
    "def detect_image_anomaly(model, img):\n",
    "    \"\"\"\n",
    "    Autoencoder + SSIM image anomaly detection.\n",
    "    \"\"\"\n",
    "\n",
    "    if img.max() > 1:\n",
    "        img = img.astype(\"float32\") / 255.0\n",
    "\n",
    "    if len(img.shape) == 2:\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)\n",
    "\n",
    "    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()\n",
    "\n",
    "    with torch.no_grad():\n",
    "        recon = model(img_t).cpu().squeeze().permute(1, 2, 0).numpy()\n",
    "\n",
    "    ssim_score, ssim_map = ssim(\n",
    "        img, recon, full=True, data_range=1.0, channel_axis=-1\n",
    "    )\n",
    "\n",
    "    anomaly_map = 1 - ssim_map\n",
    "\n",
    "    return recon, anomaly_map, ssim_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22e79556-055e-4ad8-a101-4232a0a21637",
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
