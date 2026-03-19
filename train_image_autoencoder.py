from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "image_train" / "normal"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_images(folder: Path) -> np.ndarray:
    images = []

    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
            img = imread(str(path)).astype(np.float32)

            if img.ndim == 3:
                if img.shape[-1] == 4:
                    img = img[..., :3]
                if img.shape[-1] == 3:
                    img = img.mean(axis=-1)

            if img.max() > 1.5:
                img = img / 255.0

            img = resize(
                img,
                IMG_SIZE,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)

            img = np.clip(img, 0.0, 1.0)
            img = img[..., None]
            images.append(img)

    if not images:
        raise ValueError(f"No training images found in {folder}")

    return np.stack(images, axis=0)


def build_autoencoder(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)

    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def compute_reconstruction_errors(model, x: np.ndarray) -> np.ndarray:
    recon = model.predict(x, verbose=0)
    return np.mean((x - recon) ** 2, axis=(1, 2, 3)).astype(np.float32)


def main():
    print("Loading training images...")
    x_train = load_images(TRAIN_DIR)
    print("Training images shape:", x_train.shape)

    model = build_autoencoder(input_shape=x_train.shape[1:])

    model.fit(
        x_train,
        x_train,
        epochs=20,
        batch_size=4,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )

    model_path = MODEL_DIR / "autoencoder.keras"
    model.save(model_path)
    print(f"Saved autoencoder to: {model_path}")

    errors = compute_reconstruction_errors(model, x_train)
    calib = {
        "p10": float(np.percentile(errors, 10)),
        "p90": float(np.percentile(errors, 90)),
        "n_train": int(len(errors)),
        "img_size": list(IMG_SIZE),
        "mean_error": float(np.mean(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
    }

    calib_path = MODEL_DIR / "autoencoder_calibration.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    print(f"Saved calibration to: {calib_path}")
    print("Done.")


if __name__ == "__main__":
    main()